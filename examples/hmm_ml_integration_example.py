#!/usr/bin/env python3
"""
Comprehensive Example: HMM Performance Metrics Integration with ML Models

This example demonstrates how HMM performance metrics are passed to feature generators
and integrated with ML models for improved trading predictions.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Generate realistic price data with regime changes
    returns = np.random.normal(0, 0.02, n_samples)
    
    # Add regime-specific behavior
    regime_length = n_samples // 4
    for i in range(4):
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, n_samples)
        
        if i == 0:  # Bull regime
            returns[start_idx:end_idx] += 0.001
        elif i == 1:  # Bear regime
            returns[start_idx:end_idx] -= 0.0005
        elif i == 2:  # High volatility
            returns[start_idx:end_idx] *= 2
        # i == 3: Normal regime (no change)
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def generate_base_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate basic technical features."""
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    features['volatility'] = features['returns'].rolling(20).std()
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        features[f'sma_{window}'] = data['close'].rolling(window).mean()
        features[f'price_to_sma_{window}'] = data['close'] / features[f'sma_{window}']
    
    # Volume features
    if 'volume' in data.columns:
        features['volume_sma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']
    
    # High-Low features
    if all(col in data.columns for col in ['high', 'low']):
        features['hl_ratio'] = data['high'] / data['low']
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    return features.dropna()

def example_1_basic_hmm_integration():
    """Example 1: Basic HMM performance metrics integration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic HMM Performance Metrics Integration")
    print("="*60)
    
    try:
        # Import HMM components
        from market_analysis.hmm_clustering.enhanced_hmm_clustering import (
            EnhancedHMMClustering, HMMClusteringConfig, run_hmm_clustering_analysis
        )
        
        # Create sample data
        data = create_sample_market_data(500)
        print(f"Created sample data: {data.shape}")
        
        # Configure HMM
        config = HMMClusteringConfig(
            n_components=3,
            lookback_windows=[5, 10, 20],
            technical_indicators=["rsi", "macd", "bollinger_bands"],
            use_gpu=False,
            use_memory_optimization=False,
            max_features=15
        )
        
        # Run HMM analysis
        print("Running HMM analysis...")
        result = run_hmm_clustering_analysis(
            symbol="EXAMPLE",
            interval="1h",
            config=config,
            save_results=False
        )
        
        if result is not None:
            print(f"HMM analysis completed in {result.processing_time:.2f}s")
            print(f"Performance metrics available: {len(result.performance_metrics)}")
            
            # Show key metrics
            key_metrics = ['regime_stability', 'regime_balance', 'avg_confidence', 'regime_separation_ratio']
            print("\nKey Performance Metrics:")
            for metric in key_metrics:
                if metric in result.performance_metrics:
                    print(f"  {metric}: {result.performance_metrics[metric]:.4f}")
            
            # Convert to ML features
            clustering = EnhancedHMMClustering(config)
            ml_features = clustering.get_performance_metrics_as_features(result.performance_metrics)
            print(f"\nML Features created: {ml_features.shape}")
            print(f"Feature columns: {list(ml_features.columns)[:5]}...")  # Show first 5
            
            return result, ml_features
        else:
            print("❌ HMM analysis failed")
            return None, None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This example requires the HMM clustering modules to be available")
        return None, None
    except Exception as e:
        print(f"❌ Error in basic integration: {e}")
        return None, None

def example_2_comprehensive_feature_integration():
    """Example 2: Comprehensive feature integration with base features."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Comprehensive Feature Integration")
    print("="*60)
    
    try:
        # Import integration utilities
        from src.feature_generation.utils.hmm_ml_integration import (
            HMMMLIntegrator, quick_hmm_features_integration
        )
        
        # Create sample data
        data = create_sample_market_data(800)
        print(f"Created sample data: {data.shape}")
        
        # Generate base features
        base_features = generate_base_features(data)
        print(f"Generated base features: {base_features.shape}")
        
        # Use quick integration
        print("Running quick HMM features integration...")
        integrated_features = quick_hmm_features_integration(
            data, "EXAMPLE", "1h"
        )
        
        if not integrated_features.empty:
            print(f"Integrated features: {integrated_features.shape}")
            
            # Show feature categories
            hmm_features = [col for col in integrated_features.columns if col.startswith('hmm_')]
            base_features_cols = [col for col in integrated_features.columns if not col.startswith('hmm_')]
            
            print(f"\nFeature breakdown:")
            print(f"  Base features: {len(base_features_cols)}")
            print(f"  HMM features: {len(hmm_features)}")
            print(f"  Total features: {len(integrated_features.columns)}")
            
            # Show some HMM features
            print(f"\nSample HMM features:")
            for feature in hmm_features[:10]:  # Show first 10
                print(f"  {feature}")
            
            return integrated_features
        else:
            print("❌ Feature integration failed")
            return None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This example requires the HMM ML integration modules")
        return None
    except Exception as e:
        print(f"❌ Error in comprehensive integration: {e}")
        return None

def example_3_ml_pipeline_integration():
    """Example 3: Full ML pipeline integration with ensemble weighting."""
    print("\n" + "="*60)
    print("EXAMPLE 3: ML Pipeline Integration with Ensemble Weighting")
    print("="*60)
    
    try:
        from src.feature_generation.utils.hmm_ml_integration import (
            HMMMLIntegrator, create_hmm_ensemble_pipeline
        )
        
        # Create multiple datasets for ensemble
        symbols = ["SYMBOL_A", "SYMBOL_B", "SYMBOL_C"]
        data_dict = {}
        
        for symbol in symbols:
            data_dict[symbol] = create_sample_market_data(600)
        
        print(f"Created datasets for {len(symbols)} symbols")
        
        # Create ensemble pipeline
        print("Creating HMM ensemble pipeline...")
        features_dict, ensemble_weights = create_hmm_ensemble_pipeline(
            data_dict, symbols, interval="1h"
        )
        
        if features_dict and len(ensemble_weights) > 0:
            print(f"Ensemble pipeline created successfully")
            print(f"Ensemble weights: {ensemble_weights}")
            
            # Show features for each symbol
            print(f"\nFeatures per symbol:")
            for symbol in symbols:
                if symbol in features_dict:
                    features = features_dict[symbol]
                    hmm_features = [col for col in features.columns if col.startswith('hmm_')]
                    print(f"  {symbol}: {features.shape} total, {len(hmm_features)} HMM features")
            
            # Demonstrate ensemble weighting usage
            print(f"\nEnsemble Weighting Example:")
            print(f"If you have predictions from each model:")
            sample_predictions = np.random.random((100, len(symbols)))  # 100 samples, 3 models
            
            # Weighted ensemble prediction
            ensemble_prediction = np.average(sample_predictions, axis=1, weights=ensemble_weights)
            print(f"  Individual model predictions shape: {sample_predictions.shape}")
            print(f"  Ensemble prediction shape: {ensemble_prediction.shape}")
            print(f"  Weights used: {dict(zip(symbols, ensemble_weights))}")
            
            return features_dict, ensemble_weights
        else:
            print("❌ Ensemble pipeline creation failed")
            return None, None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error in ML pipeline integration: {e}")
        return None, None

def example_4_advanced_feature_generator():
    """Example 4: Using the dedicated HMM Performance Metrics Feature Generator."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced HMM Performance Metrics Feature Generator")
    print("="*60)
    
    try:
        from src.feature_generation.categories.hmm_performance_metrics import (
            HMMPerformanceMetricsFeatureGenerator
        )
        from market_analysis.hmm_clustering.enhanced_hmm_clustering import (
            run_hmm_clustering_analysis, HMMClusteringConfig
        )
        
        # Create sample data
        data = create_sample_market_data(400)
        print(f"Created sample data: {data.shape}")
        
        # Run HMM analysis to get performance metrics
        config = HMMClusteringConfig(n_components=4, use_gpu=False, use_memory_optimization=False)
        result = run_hmm_clustering_analysis("EXAMPLE", "1h", config, save_results=False)
        
        if result is not None:
            print(f"HMM analysis completed")
            
            # Create feature generator
            feature_generator = HMMPerformanceMetricsFeatureGenerator(lookback_window=15)
            
            # Generate comprehensive features
            hmm_features = feature_generator.generate_features(
                data,
                hmm_performance_metrics=result.performance_metrics,
                regime_labels=result.regime_labels,
                regime_probabilities=result.regime_probabilities
            )
            
            print(f"Generated HMM features: {hmm_features.shape}")
            
            # Analyze feature types
            static_features = []
            dynamic_features = []
            rolling_features = []
            interaction_features = []
            
            for col in hmm_features.columns:
                if 'rolling' in col:
                    rolling_features.append(col)
                elif 'current_regime' in col or 'regime_changed' in col or 'confidence' in col:
                    dynamic_features.append(col)
                elif 'product' in col or 'score' in col or 'reliability' in col:
                    interaction_features.append(col)
                else:
                    static_features.append(col)
            
            print(f"\nFeature Analysis:")
            print(f"  Static metrics: {len(static_features)}")
            print(f"  Dynamic features: {len(dynamic_features)}")
            print(f"  Rolling features: {len(rolling_features)}")
            print(f"  Interaction features: {len(interaction_features)}")
            
            # Show sample values for different feature types
            if static_features:
                print(f"\nSample Static Features (constant across time):")
                for feature in static_features[:3]:
                    value = hmm_features[feature].iloc[0]
                    print(f"  {feature}: {value:.4f}")
            
            if dynamic_features:
                print(f"\nSample Dynamic Features (vary over time):")
                for feature in dynamic_features[:3]:
                    print(f"  {feature}: min={hmm_features[feature].min():.4f}, "
                          f"max={hmm_features[feature].max():.4f}, "
                          f"mean={hmm_features[feature].mean():.4f}")
            
            return hmm_features
        else:
            print("❌ HMM analysis failed")
            return None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in advanced feature generation: {e}")
        return None

def example_5_ml_model_training():
    """Example 5: Training ML models with HMM performance features."""
    print("\n" + "="*60)
    print("EXAMPLE 5: ML Model Training with HMM Features")
    print("="*60)
    
    try:
        # Import ML libraries (optional - graceful fallback if not available)
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            sklearn_available = True
        except ImportError:
            sklearn_available = False
            print("⚠️  scikit-learn not available, showing conceptual example")
        
        from src.feature_generation.utils.hmm_ml_integration import HMMMLIntegrator
        
        # Create sample data
        data = create_sample_market_data(1000)
        print(f"Created sample data: {data.shape}")
        
        # Prepare features using HMM integration
        integrator = HMMMLIntegrator()
        features, metadata = integrator.prepare_features_for_ml_training(
            data, "EXAMPLE", "1h", 
            base_feature_generator=generate_base_features
        )
        
        if not features.empty:
            print(f"Prepared features for ML: {features.shape}")
            print(f"Processing steps: {metadata['processing_steps']}")
            
            # Create target variable (next period return)
            target = data['close'].pct_change().shift(-1).dropna()
            
            # Align features and target
            common_index = features.index.intersection(target.index)
            features_aligned = features.loc[common_index]
            target_aligned = target.loc[common_index]
            
            print(f"Aligned data: {features_aligned.shape} features, {len(target_aligned)} targets")
            
            if sklearn_available and len(features_aligned) > 50:
                # Train ML model
                print("\nTraining Random Forest model...")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    features_aligned, target_aligned, test_size=0.2, random_state=42
                )
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print(f"Model Performance:")
                print(f"  MSE: {mse:.6f}")
                print(f"  R²: {r2:.4f}")
                
                # Feature importance analysis
                feature_importance = pd.DataFrame({
                    'feature': features_aligned.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 10 Most Important Features:")
                for idx, row in feature_importance.head(10).iterrows():
                    feature_type = "HMM" if row['feature'].startswith('hmm_') else "Base"
                    print(f"  {row['feature']} ({feature_type}): {row['importance']:.4f}")
                
                # Count HMM vs Base features in top features
                top_features = feature_importance.head(20)
                hmm_count = sum(1 for f in top_features['feature'] if f.startswith('hmm_'))
                base_count = len(top_features) - hmm_count
                
                print(f"\nTop 20 Features Breakdown:")
                print(f"  HMM features: {hmm_count}")
                print(f"  Base features: {base_count}")
                print(f"  HMM feature contribution: {hmm_count/len(top_features)*100:.1f}%")
                
                return model, feature_importance
            else:
                print("\nConceptual ML Training Example:")
                print("1. Features are prepared with HMM performance metrics")
                print("2. Target variable is created (e.g., next period returns)")
                print("3. Data is split into train/test sets")
                print("4. ML model is trained (RandomForest, XGBoost, Neural Network, etc.)")
                print("5. HMM features provide regime-aware information to improve predictions")
                print("6. Feature importance analysis shows contribution of HMM metrics")
                
                return None, None
        else:
            print("❌ Feature preparation failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error in ML model training: {e}")
        return None, None

def main():
    """Run all examples demonstrating HMM-ML integration."""
    print("HMM Performance Metrics → ML Models Integration Examples")
    print("=" * 80)
    print("This demonstrates how HMM performance metrics are passed to feature")
    print("generators and integrated with ML models for improved predictions.")
    print("=" * 80)
    
    # Run examples
    examples = [
        ("Basic Integration", example_1_basic_hmm_integration),
        ("Comprehensive Features", example_2_comprehensive_feature_integration),
        ("Ensemble Pipeline", example_3_ml_pipeline_integration),
        ("Advanced Generator", example_4_advanced_feature_generator),
        ("ML Model Training", example_5_ml_model_training)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} Running {name} {'='*20}")
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    # Summary
    print("\n" + "="*80)
    print("INTEGRATION SUMMARY")
    print("="*80)
    print("How HMM Performance Metrics Flow to ML Models:")
    print()
    print("1. HMM Analysis → Performance Metrics")
    print("   • Run HMM clustering on market data")
    print("   • Extract 25+ performance metrics (stability, balance, confidence, etc.)")
    print()
    print("2. Performance Metrics → ML Features")
    print("   • Static features: Broadcast metrics to all time points")
    print("   • Dynamic features: Regime labels, probabilities, transitions")
    print("   • Rolling features: Time-windowed metric calculations")
    print("   • Interaction features: Combined metric scores")
    print()
    print("3. Feature Integration → ML Training")
    print("   • Concatenate with base technical features")
    print("   • Apply feature selection and correlation filtering")
    print("   • Use in any ML model (RandomForest, XGBoost, Neural Networks)")
    print()
    print("4. Ensemble Weighting → Model Combination")
    print("   • Weight models based on HMM performance metrics")
    print("   • Higher weight for models with better regime detection")
    print("   • Combine predictions using performance-based weights")
    print()
    print("Key Benefits:")
    print("• Regime-aware predictions")
    print("• Model quality assessment")
    print("• Automatic ensemble weighting")
    print("• Enhanced feature engineering")
    print("• Meta-learning capabilities")
    
    successful_examples = sum(1 for result in results.values() if result is not None)
    print(f"\nResults: {successful_examples}/{len(examples)} examples completed successfully")

if __name__ == "__main__":
    main()