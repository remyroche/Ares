"""
Enhanced Artifact Manager Usage Example

This example demonstrates how to use the enhanced artifact manager with:
- Proper context setting (symbol, exchange, datetime, information)
- Enhanced file naming with information + symbol + exchange + datetime
- Full path logging for all operations
- Joint Parquet file creation
- JSON metadata generation
- Data alignment verification
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactConfig,
    ArtifactKeys
)

def example_enhanced_artifact_manager_usage():
    """Example of enhanced artifact manager usage."""
    
    # Initialize artifact manager with enhanced configuration
    config = ArtifactConfig(
        include_symbol_in_filename=True,
        include_exchange_in_filename=True,
        include_datetime_in_filename=True,
        include_information_in_filename=True,
        use_joint_parquet_format=True,
        generate_json_metadata=True
    )
    
    am = get_pretraining_artifact_manager()
    am.config = config
    
    # Set context for enhanced file naming
    am.set_context(
        symbol="ETHUSDT",
        exchange="binance",
        datetime=datetime(2024, 1, 15, 10, 30, 0),
        information="pre_training"
    )
    
    print("🚀 Enhanced Artifact Manager Example")
    print("=" * 50)
    
    # Example 1: Basic artifact saving with enhanced naming
    print("\n📁 Example 1: Basic artifact saving")
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    ohlcv_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Save OHLCV data
    am.save(
        step_name='feature_generation_data_validation_step',
        artifacts={
            'raw_dataframe': ohlcv_data,
            'validation_metrics': {'quality_score': 0.95, 'completeness': 0.98}
        },
        metadata={'step_info': 'Data validation completed'}
    )
    
    # Example 2: Feature generation with enhanced naming
    print("\n📁 Example 2: Feature generation")
    
    # Create sample features
    features_data = pd.DataFrame({
        'sma_20': ohlcv_data['close'].rolling(20).mean(),
        'rsi_14': np.random.uniform(20, 80, 100),
        'bb_upper': ohlcv_data['close'].rolling(20).mean() + 2 * ohlcv_data['close'].rolling(20).std(),
        'bb_lower': ohlcv_data['close'].rolling(20).mean() - 2 * ohlcv_data['close'].rolling(20).std()
    }, index=dates)
    
    feature_names = list(features_data.columns)
    feature_categories = {
        'technical_indicators': ['sma_20', 'rsi_14'],
        'bollinger_bands': ['bb_upper', 'bb_lower']
    }
    
    am.save(
        step_name='feature_generation_feature_generation_step',
        artifacts={
            'feature_dataframe': features_data,
            'feature_names': feature_names,
            'feature_categories': feature_categories,
            'generation_metrics': {'total_features': len(feature_names), 'generation_time': 2.5}
        },
        metadata={'generation_method': 'technical_analysis'}
    )
    
    # Example 3: Joint Parquet file creation
    print("\n📁 Example 3: Joint Parquet file creation")
    
    # Create sample labels
    labels_data = pd.DataFrame({
        'target_1h': np.random.choice([-1, 0, 1], 100),
        'target_4h': np.random.choice([-1, 0, 1], 100),
        'profit_label': np.random.uniform(-0.05, 0.05, 100)
    }, index=dates)
    
    # Create joint Parquet file
    joint_path = am.create_joint_parquet_file(
        step_name='feature_generation_final_validation_step',
        ohlcv_data=ohlcv_data,
        labels_data=labels_data,
        features_data=features_data,
        key='final_dataset'
    )
    
    print(f"✅ Joint Parquet file created: {joint_path}")
    
    # Example 4: Retrieving artifacts with full path logging
    print("\n📁 Example 4: Retrieving artifacts")
    
    # Retrieve OHLCV data
    retrieved_ohlcv = am.get_artifact('feature_generation_data_validation_step', 'raw_dataframe')
    print(f"Retrieved OHLCV data: {type(retrieved_ohlcv)} with shape {retrieved_ohlcv.shape if hasattr(retrieved_ohlcv, 'shape') else 'N/A'}")
    
    # Retrieve features
    retrieved_features = am.get_artifact('feature_generation_feature_generation_step', 'feature_dataframe')
    print(f"Retrieved features: {type(retrieved_features)} with shape {retrieved_features.shape if hasattr(retrieved_features, 'shape') else 'N/A'}")
    
    # Example 5: Data alignment verification
    print("\n📁 Example 5: Data alignment verification")
    
    # The joint file creation automatically verifies alignment
    print("✅ Data alignment verified during joint file creation")
    
    # Example 6: JSON metadata generation
    print("\n📁 Example 6: JSON metadata generation")
    
    # The artifact manager automatically generates JSON metadata for feature-related steps
    print("✅ JSON metadata automatically generated for feature steps")
    
    # Example 7: Enhanced path structure
    print("\n📁 Example 7: Enhanced path structure")
    
    # Show the enhanced directory structure
    base_dir = am.config.base_dir
    print(f"Base directory: {base_dir}")
    
    # List created files
    for file_path in base_dir.rglob("*"):
        if file_path.is_file():
            print(f"📄 {file_path}")
    
    print("\n🎉 Enhanced Artifact Manager example completed!")
    print("=" * 50)
    
    # Show metrics
    metrics = am.get_metrics()
    print(f"\n📊 Metrics: {metrics}")

if __name__ == "__main__":
    example_enhanced_artifact_manager_usage()
