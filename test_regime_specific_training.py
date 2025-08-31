#!/usr/bin/env python3
"""
Test script for Regime-Specific ML Model Training

This script demonstrates how to train ML models specifically on data from each HMM cluster/regime.
Each model is trained only on the data that belongs to its specific HMM state.
"""

import asyncio
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.regime_specific_ml_trainer import run_regime_specific_training


def create_test_config():
    """Create test configuration for regime-specific training."""
    config = {
        "regime_specific_training": {
            "hmm_data_path": "data/training/hmm_regimes",
            "model_output_path": "models/regime_specific",
            "regime_models": [
                "exit_timing",
                "exit_probabilities", 
                "exit_type",
                "profit_target"
            ],
            "training_parameters": {
                "test_size": 0.2,
                "random_state": 42,
                "min_samples_per_regime": 1000,
                "feature_selection": True,
                "cross_validation": True
            }
        },
        "model_config": {
            "lightgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            },
            "catboost": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "random_state": 42
            }
        }
    }
    return config


async def test_regime_specific_training():
    """Test regime-specific ML model training."""
    print("🚀 Testing Regime-Specific ML Model Training")
    print("=" * 60)
    
    # Create configuration
    config = create_test_config()
    
    # Test parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    force_retrain = True
    
    print(f"🎯 Symbol: {symbol}")
    print(f"🏢 Exchange: {exchange}")
    print(f"📊 Timeframe: {timeframe}")
    print(f"🔄 Force retrain: {force_retrain}")
    print()
    
    # Run training
    print("🔄 Starting regime-specific model training...")
    success = await run_regime_specific_training(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        config=config,
        force_retrain=force_retrain
    )
    
    if success:
        print("✅ Regime-specific training completed successfully!")
        
        # Show results
        await show_training_results(config)
    else:
        print("❌ Regime-specific training failed!")
    
    print("=" * 60)


async def show_training_results(config):
    """Show training results."""
    try:
        results_path = Path(config["regime_specific_training"]["model_output_path"]) / "training_results.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("\n📊 Training Results:")
            print(f"   📅 Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"   🎯 Symbol: {results.get('symbol', 'N/A')}")
            print(f"   🏢 Exchange: {results.get('exchange', 'N/A')}")
            print(f"   📊 Timeframe: {results.get('timeframe', 'N/A')}")
            print(f"   🔄 Regimes trained: {results.get('regimes_trained', [])}")
            print(f"   📈 Total models: {results.get('total_models', 0)}")
            
            # Show training summary
            summary = results.get('training_summary', {})
            if summary:
                print("\n📋 Training Summary:")
                for regime_id, regime_info in summary.items():
                    print(f"   Regime {regime_id}:")
                    print(f"     - Models: {regime_info.get('models_trained', [])}")
                    print(f"     - Count: {regime_info.get('model_count', 0)}")
        
        # Show model files
        model_path = Path(config["regime_specific_training"]["model_output_path"])
        if model_path.exists():
            print(f"\n📁 Model files saved to: {model_path}")
            
            for regime_dir in model_path.iterdir():
                if regime_dir.is_dir() and regime_dir.name != "__pycache__":
                    print(f"   📂 Regime {regime_dir.name}:")
                    for model_file in regime_dir.glob("*.pkl"):
                        print(f"     - {model_file.name}")
                    for json_file in regime_dir.glob("*.json"):
                        print(f"     - {json_file.name}")
    
    except Exception as e:
        print(f"❌ Error showing training results: {e}")


def explain_training_process():
    """Explain the regime-specific training process."""
    print("\n📚 Regime-Specific Training Process Explanation:")
    print("=" * 60)
    
    print("""
🎯 **How Regime-Specific Training Works:**

1. **Load HMM Regime Data**
   - Load the parquet file from step3 HMM regime discovery
   - File: data/training/hmm_regimes/BINANCE_ETHUSDT_hmm_block_states_1m.parquet
   - Contains: composite_cluster_id, features, timestamp

2. **Split Data by Regime**
   - Filter data for each unique HMM regime/cluster
   - Each regime gets its own training dataset
   - Example: Regime 0, Regime 1, Regime 2, etc.

3. **Create Regime-Specific Targets**
   - Exit Timing: How many bars to hold (regression)
   - Exit Probabilities: Probability of exit within X bars (classification)
   - Exit Type: Type of exit (take_profit, stop_loss, etc.)
   - Profit Target: Optimal profit percentage (regression)

4. **Train Models for Each Regime**
   - Each regime gets 4 specialized models
   - Models learn from only that regime's data
   - Different algorithms: LightGBM, XGBoost, CatBoost

5. **Save Regime-Specific Models**
   - Models saved in: models/regime_specific/{regime_id}/
   - Each model includes: model.pkl, scaler.pkl, metrics.json

🎯 **Training Goals for Each Regime:**

**High Volatility Regimes:**
- Quick exits (1-10 bars)
- Higher exit probabilities
- More stop losses

**Low Volatility Regimes:**
- Longer holds (20-50 bars)
- Lower exit probabilities
- More take profits

**Bull Market Regimes:**
- Trend following
- Medium-term holds
- Profit taking focus

**Bear Market Regimes:**
- Defensive exits
- Quick stops
- Risk management focus
""")


if __name__ == "__main__":
    print("🧪 Regime-Specific ML Training Test")
    print("=" * 60)
    
    # Explain the process
    explain_training_process()
    
    # Run test
    asyncio.run(test_regime_specific_training())