"""
Example: Artifact Versioning and Pickup System

This example demonstrates how to use the enhanced artifact management system
with version and timestamp support, and automatic pickup of the most recent artifacts.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Import the enhanced artifact management system
from src.utils.enhanced_artifact_manager import get_artifact_manager, initialize_artifact_manager
from src.utils.version_manager import get_version_manager, set_ares_version
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils


async def main():
    """Main example function demonstrating artifact versioning and pickup."""
    
    print("🚀 Ares Artifact Versioning and Pickup Example")
    print("=" * 50)
    
    # 1. Initialize the system
    print("\n1. Initializing Artifact Management System")
    
    # Configure the artifact manager
    config = {
        "ares_version": "v1",
        "data_dir": "data",
        "model_dir": "models", 
        "artifacts_dir": "artifacts",
        "cache_dir": "data_cache",
        "output_dir": "output"
    }
    
    # Initialize managers
    artifact_manager = initialize_artifact_manager(config)
    version_manager = get_version_manager()
    pickup_utils = get_artifact_pickup_utils()
    
    print(f"✅ Initialized with Ares version: {version_manager.get_ares_version()}")
    
    # 2. Simulate creating artifacts in different pipeline stages
    print("\n2. Creating Artifacts in Pipeline Stages")
    
    # Stage 1: Data Collection - Create sample data
    print("\n📥 Stage 1: Data Collection")
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1000, freq='1min'),
        'price': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 1000)
    })
    
    # Save with versioned filename
    data_file = artifact_manager.save_artifact(
        sample_data,
        "collected_data",
        ".parquet",
        "artifacts"
    )
    print(f"✅ Saved data collection artifact: {Path(data_file).name}")
    
    # Stage 2: Market Analysis - Create processed features
    print("\n📊 Stage 2: Market Analysis")
    features = pd.DataFrame({
        'timestamp': sample_data['timestamp'],
        'sma_20': sample_data['price'].rolling(20).mean(),
        'rsi': np.random.uniform(0, 100, 1000),
        'volatility': sample_data['price'].rolling(20).std()
    }).dropna()
    
    # Save features
    features_file = artifact_manager.save_artifact(
        features,
        "market_features",
        ".parquet", 
        "artifacts"
    )
    print(f"✅ Saved market analysis artifact: {Path(features_file).name}")
    
    # Stage 3: Model Training - Create a simple model
    print("\n🤖 Stage 3: Model Training")
    from sklearn.linear_model import LinearRegression
    
    # Create a simple model
    model = LinearRegression()
    X = features[['sma_20', 'rsi', 'volatility']].values
    y = features['sma_20'].shift(-1).dropna().values
    X = X[:-1]  # Align with y
    
    model.fit(X, y)
    
    # Save model
    model_file = artifact_manager.save_artifact(
        model,
        "trained_model",
        ".pkl",
        "artifacts"
    )
    print(f"✅ Saved model artifact: {Path(model_file).name}")
    
    # 3. Demonstrate artifact pickup
    print("\n3. Demonstrating Artifact Pickup")
    
    # Find most recent data collection artifact
    print("\n🔍 Finding Most Recent Data Collection Artifact")
    recent_data_path = pickup_utils.find_most_recent_artifact("collected_data", "artifacts")
    if recent_data_path:
        print(f"✅ Found most recent data artifact: {Path(recent_data_path).name}")
        
        # Load the data
        loaded_data, metadata = pickup_utils.load_most_recent_artifact("collected_data", "artifacts")
        if loaded_data is not None:
            print(f"✅ Loaded data with shape: {loaded_data.shape}")
            print(f"   Version: {metadata.version}")
            print(f"   Timestamp: {metadata.timestamp}")
    
    # Find most recent market analysis artifact
    print("\n🔍 Finding Most Recent Market Analysis Artifact")
    recent_features_path = pickup_utils.find_most_recent_artifact("market_features", "artifacts")
    if recent_features_path:
        print(f"✅ Found most recent features artifact: {Path(recent_features_path).name}")
    
    # Find most recent model artifact
    print("\n🔍 Finding Most Recent Model Artifact")
    recent_model_path = pickup_utils.find_most_recent_artifact("trained_model", "artifacts")
    if recent_model_path:
        print(f"✅ Found most recent model artifact: {Path(recent_model_path).name}")
    
    # 4. Demonstrate version management
    print("\n4. Demonstrating Version Management")
    
    # Change version and create new artifacts
    print("\n🔄 Changing Ares Version to v2")
    set_ares_version("v2")
    print(f"✅ New Ares version: {version_manager.get_ares_version()}")
    
    # Create new artifacts with new version
    print("\n📊 Creating New Artifacts with v2")
    new_features = features.copy()
    new_features['new_feature'] = np.random.randn(len(new_features))
    
    new_features_file = artifact_manager.save_artifact(
        new_features,
        "market_features",
        ".parquet",
        "artifacts"
    )
    print(f"✅ Saved v2 features artifact: {Path(new_features_file).name}")
    
    # 5. List all available artifacts
    print("\n5. Listing All Available Artifacts")
    
    all_artifacts = pickup_utils.list_available_artifacts("artifacts")
    for base_name, artifacts in all_artifacts.items():
        print(f"\n📁 {base_name}:")
        for artifact in artifacts:
            version = artifact.get('version', 'unknown')
            timestamp = artifact.get('timestamp', 'unknown')
            size = artifact.get('size_bytes', 0)
            print(f"   - {artifact['filename']} (v{version}, {timestamp}, {size} bytes)")
    
    # 6. Demonstrate cleanup
    print("\n6. Demonstrating Artifact Cleanup")
    
    # Create multiple versions of the same artifact for cleanup demo
    print("\n📝 Creating Multiple Versions for Cleanup Demo")
    for i in range(3):
        test_data = pd.DataFrame({'test': [i] * 100})
        artifact_manager.save_artifact(
            test_data,
            "cleanup_test",
            ".parquet",
            "artifacts"
        )
        print(f"   Created cleanup_test artifact {i+1}")
    
    # List before cleanup
    cleanup_artifacts = pickup_utils.find_artifacts_by_pattern("cleanup_test_*", "artifacts")
    print(f"\n📊 Before cleanup: {len(cleanup_artifacts)} cleanup_test artifacts")
    
    # Cleanup (keep only 2 most recent)
    deleted_files = pickup_utils.cleanup_old_artifacts("cleanup_test", "artifacts", keep_count=2)
    print(f"🗑️ Cleaned up {len(deleted_files)} old artifacts")
    
    # List after cleanup
    cleanup_artifacts_after = pickup_utils.find_artifacts_by_pattern("cleanup_test_*", "artifacts")
    print(f"📊 After cleanup: {len(cleanup_artifacts_after)} cleanup_test artifacts")
    
    # 7. Demonstrate pipeline artifact discovery
    print("\n7. Demonstrating Pipeline Artifact Discovery")
    
    # Get artifacts for a specific pipeline stage
    pipeline_artifacts = pickup_utils.get_pipeline_artifacts(
        "market_analysis",
        ["features", "labels", "metadata"],
        "artifacts"
    )
    
    print("\n📋 Pipeline Artifacts for 'market_analysis':")
    for artifact_type, file_path in pipeline_artifacts.items():
        if file_path:
            print(f"   ✅ {artifact_type}: {Path(file_path).name}")
        else:
            print(f"   ❌ {artifact_type}: Not found")
    
    print("\n🎉 Example completed successfully!")
    print("\nKey Benefits:")
    print("✅ Automatic version and timestamp in filenames")
    print("✅ Easy discovery of most recent artifacts")
    print("✅ Version-aware artifact management")
    print("✅ Automatic cleanup of old artifacts")
    print("✅ Pipeline-aware artifact organization")


if __name__ == "__main__":
    asyncio.run(main())