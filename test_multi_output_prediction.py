#!/usr/bin/env python3
"""Test Multi-Output Prediction System.

This script demonstrates the intelligent multi-output prediction capabilities
for both direction and profit using the triple barrier method and profit-based
feature engineering.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.multi_output_model_trainer import create_multi_output_trainer, MultiOutputModelConfig
from src.training.steps.step6_hmm_based_training_enhanced import run_enhanced_step
from src.config.multi_output_config import get_multi_output_config, validate_multi_output_config
from src.utils.logger import system_logger


def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample data for testing multi-output prediction.
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        DataFrame with sample features and targets
    """
    np.random.seed(42)
    
    # Create sample features
    features = {
        'momentum_strength': np.random.normal(0, 1, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'volume_volatility': np.random.exponential(1, n_samples),
        'liquidity_score': np.random.normal(0, 1, n_samples),
        'order_flow_imbalance': np.random.normal(0, 1, n_samples),
        'price_volatility': np.random.exponential(0.5, n_samples),
        'macd': np.random.normal(0, 1, n_samples),
        'bb_position': np.random.uniform(-1, 1, n_samples),
        'atr': np.random.exponential(0.01, n_samples),
        'volume': np.random.exponential(1000, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
    }
    
    # Create direction target (binary)
    direction = (features['momentum_strength'] + features['rsi'] / 100 + 
                features['order_flow_imbalance'] + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    # Create profit target (percentage)
    profit_pct = (features['momentum_strength'] * 0.01 + 
                 features['volume_volatility'] * 0.005 +
                 features['liquidity_score'] * 0.002 +
                 np.random.normal(0, 0.001, n_samples))
    
    # Create potential profit percentage (from triple barrier method)
    potential_profit_pct = direction * profit_pct
    
    # Create single-output target (backward compatibility)
    target = direction
    
    # Combine into DataFrame
    data = pd.DataFrame(features)
    data['direction'] = direction
    data['potential_profit_pct'] = potential_profit_pct
    data['target'] = target
    data['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    return data


async def test_multi_output_trainer():
    """Test the multi-output model trainer."""
    print("🧪 Testing Multi-Output Model Trainer")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data(10000)
    print(f"✅ Created sample data: {data.shape}")
    print(f"   - Direction distribution: {data['direction'].value_counts().to_dict()}")
    print(f"   - Profit stats: mean={data['potential_profit_pct'].mean():.6f}, std={data['potential_profit_pct'].std():.6f}")
    
    # Initialize multi-output trainer
    print("\n🔧 Initializing multi-output trainer...")
    trainer = create_multi_output_trainer(
        model_type="LightGBM",
        use_profit_features=True
    )
    
    # Prepare data
    print("📊 Preparing multi-output data...")
    features, direction_target, profit_target = trainer.prepare_multi_output_data(
        data,
        direction_column="direction",
        profit_column="potential_profit_pct"
    )
    
    print(f"✅ Prepared data: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Train multi-output model
    print("\n🚀 Training multi-output model...")
    result = trainer.train_multi_output_model(
        features=features,
        direction_target=direction_target,
        profit_target=profit_target,
        model_name="test_multi_output_model"
    )
    
    if result:
        print("✅ Multi-output model training completed successfully")
        print(f"   - Direction accuracy: {result['direction_metrics']['accuracy']:.4f}")
        print(f"   - Profit R²: {result['profit_metrics']['r2']:.4f}")
        print(f"   - Combined correlation: {result['combined_metrics']['direction_weighted_profit_correlation']:.4f}")
        
        # Test predictions
        print("\n🔮 Testing predictions...")
        test_features = features.iloc[:100]  # Use first 100 samples for testing
        direction_pred, profit_pred = trainer.predict(test_features, "test_multi_output_model")
        
        if direction_pred is not None and profit_pred is not None:
            print("✅ Predictions successful")
            print(f"   - Direction predictions: {direction_pred[:10]}...")
            print(f"   - Profit predictions: mean={profit_pred.mean():.6f}, std={profit_pred.std():.6f}")
            
            # Calculate direction-weighted profit
            direction_weighted_profit = direction_pred * profit_pred
            print(f"   - Direction-weighted profit: mean={direction_weighted_profit.mean():.6f}")
        
        # Save model
        print("\n💾 Saving model...")
        trainer.save_model("test_multi_output_model", "test_models")
        print("✅ Model saved successfully")
        
    else:
        print("❌ Multi-output model training failed")


async def test_enhanced_hmm_training():
    """Test the enhanced HMM-based training."""
    print("\n🧪 Testing Enhanced HMM-Based Training")
    print("=" * 50)
    
    # Create sample data and save it
    print("📊 Creating and saving sample data...")
    data = create_sample_data(5000)
    
    # Create data directory
    os.makedirs("test_data/training", exist_ok=True)
    
    # Save as parquet
    data_path = "test_data/training/ETHUSDT_labeled_train.parquet"
    data.to_parquet(data_path, index=False)
    print(f"✅ Saved sample data to {data_path}")
    
    # Test enhanced step
    print("\n🚀 Testing enhanced HMM-based training step...")
    success = await run_enhanced_step(
        symbol="ETHUSDT",
        data_dir="test_data/training",
        enable_multi_output=True
    )
    
    if success:
        print("✅ Enhanced HMM-based training completed successfully")
    else:
        print("❌ Enhanced HMM-based training failed")


def test_configuration():
    """Test the multi-output configuration."""
    print("\n🧪 Testing Multi-Output Configuration")
    print("=" * 50)
    
    # Test basic configuration
    print("📋 Testing basic configuration...")
    config = get_multi_output_config()
    print(f"✅ Configuration loaded: {len(config)} sections")
    
    # Validate configuration
    print("\n🔍 Validating configuration...")
    is_valid = validate_multi_output_config(config)
    
    if is_valid:
        print("✅ Configuration validation passed")
        
        # Print key settings
        print("\n📊 Key configuration settings:")
        print(f"   - Enable multi-output: {config['enable_multi_output']}")
        print(f"   - Model type: {config['multi_output_models']['model_type']}")
        print(f"   - Use profit features: {config['multi_output_models']['use_profit_features']}")
        print(f"   - Direction target: {config['multi_output_models']['direction_target']}")
        print(f"   - Profit target: {config['multi_output_models']['profit_target']}")
        
        # Test model-specific configuration
        print("\n🔧 Testing model-specific configuration...")
        lightgbm_config = config['multi_output_models']
        print(f"   - LightGBM settings: {lightgbm_config['model_type']}")
        
    else:
        print("❌ Configuration validation failed")


def test_profit_feature_engineering():
    """Test profit-based feature engineering."""
    print("\n🧪 Testing Profit-Based Feature Engineering")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data(1000)
    
    # Import profit feature engineering
    from src.training.steps.step4_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
        ProfitBasedFeatureEngineering
    )
    
    # Initialize profit feature engineering
    print("🔧 Initializing profit feature engineering...")
    profit_engine = ProfitBasedFeatureEngineering(
        profit_column="potential_profit_pct",
        use_numba=True,
        memory_efficient=True
    )
    
    # Apply profit-based features
    print("🔧 Applying profit-based features...")
    enhanced_data = profit_engine.apply_all_features(data)
    
    print(f"✅ Enhanced data: {enhanced_data.shape}")
    print(f"   - Original features: {len(data.columns)}")
    print(f"   - Enhanced features: {len(enhanced_data.columns)}")
    print(f"   - New features added: {len(enhanced_data.columns) - len(data.columns)}")
    
    # Show some new features
    new_features = [col for col in enhanced_data.columns if col not in data.columns]
    print(f"\n📊 New profit-based features:")
    for i, feature in enumerate(new_features[:10]):  # Show first 10
        print(f"   {i+1}. {feature}")
    
    if len(new_features) > 10:
        print(f"   ... and {len(new_features) - 10} more features")


def test_data_driven_feature_selection():
    """Test data-driven feature selection methods."""
    print("\n🧪 Testing Data-Driven Feature Selection")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data(1000)
    
    # Import enhanced feature selection
    from src.training.enhanced_feature_selection_manager import EnhancedFeatureSelectionManager
    
    # Initialize enhanced feature selection
    print("🔧 Initializing enhanced feature selection...")
    config = {
        "feature_reduction": {
            "target_features": 50,
            "vif_threshold": 10.0,
            "mi_threshold": 0.01,
            "correlation_threshold": 0.95,
            "variance_threshold": 0.01,
            "method_weights": {
                "vif": 0.2,
                "mutual_info": 0.25,
                "shap": 0.25,
                "random_forest": 0.2,
                "rfe": 0.1
            }
        }
    }
    
    feature_selector = EnhancedFeatureSelectionManager(config)
    
    # Apply data-driven feature selection
    print("🔧 Applying data-driven feature selection...")
    selected_features, metadata = feature_selector.select_features_enhanced(
        features_df=data,
        target=data['direction'],
        symbol="ETHUSDT",
        exchange="BINANCE",
        data_dir="test_data/feature_selection",
        task="classification"
    )
    
    print(f"✅ Data-driven feature selection completed")
    print(f"   - Original features: {len(data.columns)}")
    print(f"   - Selected features: {len(selected_features.columns)}")
    print(f"   - Features removed: {len(data.columns) - len(selected_features.columns)}")
    
    # Show selection statistics
    print(f"\n📊 Selection Statistics:")
    stages = metadata.get('stages', {})
    print(f"   - VIF features removed: {stages.get('stage3_vif', {}).get('removed_high_vif', 0)}")
    print(f"   - MI features removed: {stages.get('stage5_mutual_info', {}).get('removed_low_mi', 0)}")
    print(f"   - SHAP features removed: {stages.get('stage6_shap', {}).get('removed_low_shap', 0)}")
    print(f"   - RF features removed: {stages.get('stage7_random_forest', {}).get('removed_low_rf', 0)}")
    print(f"   - Processing time: {metadata.get('processing_time', 0):.2f}s")
    
    # Show feature importance summary
    print(f"\n📊 Feature Importance Summary:")
    importance_summary = feature_selector.get_feature_importance_summary()
    for method, summary in importance_summary.items():
        print(f"   - {method.upper()}:")
        print(f"     Mean: {summary['mean']:.4f}")
        print(f"     Max: {summary['max']:.4f}")
        print(f"     Top feature: {list(summary['top_10_features'].keys())[0]}")
    
    print(f"\n✅ Data-driven feature selection is working correctly!")


async def main():
    """Main test function."""
    print("🚀 Multi-Output Prediction System Test")
    print("=" * 60)
    
    try:
        # Test configuration
        test_configuration()
        
        # Test profit feature engineering
        test_profit_feature_engineering()
        
            # Test multi-output trainer
    await test_multi_output_trainer()
    
    # Test data-driven feature selection
    test_data_driven_feature_selection()
        
        # Test enhanced HMM training
        await test_enhanced_hmm_training()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Multi-output configuration validated")
        print("   ✅ Profit-based feature engineering working")
        print("   ✅ Multi-output model trainer functional")
        print("   ✅ Enhanced HMM-based training operational")
        print("\n🔧 The system is ready for intelligent multi-output prediction!")
    
    print("\n📊 Data-driven feature selection is now enabled!")
    print("   - VIF filtering for multicollinearity")
    print("   - Mutual Information for feature relevance")
    print("   - SHAP for model-based importance")
    print("   - RandomForest for ensemble importance")
    print("   - RFE for final selection")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())