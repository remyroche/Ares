#!/usr/bin/env python3
"""
Test script for Ray-based model trainer.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.model_trainer import setup_model_trainer


def test_ray_model_trainer():
    """Test the Ray-based model trainer."""
    
    # Configuration for testing
    config = {
        "ray": {
            "num_cpus": 2,  # Use fewer CPUs for testing
            "num_gpus": 0,
            "logging_level": "info"
        },
        "model_trainer": {
            "enable_analyst_models": True,
            "enable_tactician_models": True,
            "model_directory": "test_models",
            "analyst_models": {
                "timeframes": ["1h", "15m", "5m", "1m"]
            },
            "tactician_models": {
                "timeframes": ["1m"]
            }
        }
    }
    
    print("🚀 Testing Ray-based Model Trainer...")
    
    # Setup trainer
    trainer = setup_model_trainer(config)
    
    if not trainer:
        print("❌ Failed to setup model trainer!")
        return False
    
    print("✅ Model trainer setup successful!")
    
    # Check training status
    status = trainer.get_training_status()
    print(f"📊 Training status: {status}")
    
    # Example training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "lookback_days": 30
    }
    
    print("🧠 Starting model training...")
    
    try:
        # Train models
        results = trainer.train_models(training_input)
        
        if results:
            print("✅ Training completed successfully!")
            
            # Print results summary
            analyst_models = results.get('analyst_models', {})
            tactician_models = results.get('tactician_models', {})
            
            print(f"📈 Analyst models trained: {len(analyst_models)}")
            for timeframe, result in analyst_models.items():
                if result['training_status'] == 'completed':
                    metrics = result['model_metrics']
                    print(f"  - {timeframe}: Accuracy={metrics['accuracy']:.3f}, "
                          f"Precision={metrics['precision']:.3f}, "
                          f"Recall={metrics['recall']:.3f}")
            
            print(f"🎯 Tactician models trained: {len(tactician_models)}")
            for timeframe, result in tactician_models.items():
                if result['training_status'] == 'completed':
                    metrics = result['model_metrics']
                    print(f"  - {timeframe}: Accuracy={metrics['accuracy']:.3f}, "
                          f"Precision={metrics['precision']:.3f}, "
                          f"Recall={metrics['recall']:.3f}")
            
            # Test model loading
            print("\n🔍 Testing model loading...")
            for model_type in ['analyst', 'tactician']:
                for timeframe in ['1m']:
                    model_data = trainer.load_model(model_type, timeframe)
                    if model_data:
                        model, scaler = model_data
                        print(f"✅ Successfully loaded {model_type}_{timeframe} model")
                    else:
                        print(f"❌ Failed to load {model_type}_{timeframe} model")
            
            return True
        else:
            print("❌ Training failed!")
            return False
    
    finally:
        # Cleanup
        print("\n🛑 Cleaning up...")
        trainer.stop()
        print("✅ Cleanup completed!")


if __name__ == "__main__":
    success = test_ray_model_trainer()
    sys.exit(0 if success else 1)