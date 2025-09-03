#!/usr/bin/env python3
"""Update pipeline configuration for new Step 7 and Step 8."""

import json
import os
from pathlib import Path


def update_training_config():
    """Update training configuration for new feature selection steps."""
    
    # Default configuration for new steps
    step07_config = {
        "step07_enhanced_matrix_operations": {
            "output_dir": "data/matrix_operations",
            "target_features": 200,
            "removal_fraction": 0.33,
            "enable_regime_selection": True,
            "enable_shap_filtering": True
        }
    }
    
    step08_config = {
        "step08_advanced_feature_selection": {
            "output_dir": "data/selected_features",
            "phase1_target_features": 150,
            "enable_mrmr": True,
            "enable_rf_importance": True,
            "phase2_targets": [100, 80, 60],
            "boruta_max_iter": 100,
            "boruta_alpha": 0.05,
            "n_splits_ts": 5,
            "min_regime_samples": 100,
            "enable_shap": True,
            "enable_lime": True,
            "n_lime_samples": 10
        }
    }
    
    # Load existing config if available
    config_path = "config/training_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update with new step configurations
    config.update(step07_config)
    config.update(step08_config)
    
    # Save updated config
    os.makedirs("config", exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated training configuration saved to {config_path}")


def update_step09_for_new_features():
    """Provide instructions for updating Step 9 to use Step 8 outputs."""
    
    print("\n" + "="*60)
    print("📝 INSTRUCTIONS FOR UPDATING STEP 9")
    print("="*60)
    print("""
Step 9 (HMM-Based Training) should be updated to use the feature sets from Step 8:

1. Check for Step 8 outputs first:
   - data/selected_features/{exchange}_{symbol}_{timeframe}_top100_train.parquet
   - data/selected_features/{exchange}_{symbol}_{timeframe}_top100_val.parquet

2. If Step 8 outputs exist, use them. Otherwise, fall back to:
   - Step 7 outputs (filtered features)
   - Original Step 6 outputs (all features)

3. Example code to add to Step 9:

```python
# Try to load Step 8 outputs (advanced selection)
step8_train_path = f"data/selected_features/{exchange}_{symbol}_{timeframe}_top100_train.parquet"
step8_val_path = f"data/selected_features/{exchange}_{symbol}_{timeframe}_top100_val.parquet"

if os.path.exists(step8_train_path) and os.path.exists(step8_val_path):
    logger.info("✅ Using Step 8 advanced feature selection outputs")
    train_data = pd.read_parquet(step8_train_path)
    val_data = pd.read_parquet(step8_val_path)
else:
    # Try Step 7 outputs (filtered features)
    step7_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet"
    step7_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet"
    
    if os.path.exists(step7_train_path) and os.path.exists(step7_val_path):
        logger.info("✅ Using Step 7 filtered features")
        train_data = pd.read_parquet(step7_train_path)
        val_data = pd.read_parquet(step7_val_path)
    else:
        # Fall back to original features
        logger.warning("⚠️ Using original features (no feature selection applied)")
        train_data = pd.read_parquet(original_train_path)
        val_data = pd.read_parquet(original_val_path)
```

4. For different model types, use different feature sets:
   - Neural networks: top80 features
   - Linear models: top60 features
   - Ensemble models: top100 features
""")


def create_pipeline_runner():
    """Create a simple pipeline runner script."""
    
    runner_content = '''#!/usr/bin/env python3
"""Run the complete feature engineering and selection pipeline."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.step06_feature_engineering import run_step as run_step06
from src.training.steps.step07_enhanced_matrix_operations import run_step as run_step07
from src.training.steps.step08_advanced_feature_selection import run_step as run_step08
from src.utils.logger import system_logger


async def run_feature_pipeline(symbol: str, exchange: str, timeframe: str = "1m"):
    """Run the complete feature engineering and selection pipeline."""
    
    logger = system_logger.getChild("FeaturePipeline")
    
    # Step 6: Feature Engineering
    logger.info("Running Step 6: Feature Engineering...")
    if not await run_step06(symbol, exchange, timeframe):
        logger.error("Step 6 failed!")
        return False
    
    # Step 7: Matrix Operations & Initial Filtering
    logger.info("Running Step 7: Matrix Operations & Filtering...")
    if not await run_step07(symbol, exchange, timeframe):
        logger.error("Step 7 failed!")
        return False
    
    # Step 8: Advanced Feature Selection
    logger.info("Running Step 8: Advanced Feature Selection...")
    if not await run_step08(symbol, exchange, timeframe):
        logger.error("Step 8 failed!")
        return False
    
    logger.info("✅ Feature pipeline completed successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_feature_pipeline.py <symbol> <exchange> [timeframe]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    exchange = sys.argv[2]
    timeframe = sys.argv[3] if len(sys.argv) > 3 else "1m"
    
    success = asyncio.run(run_feature_pipeline(symbol, exchange, timeframe))
    sys.exit(0 if success else 1)
'''
    
    runner_path = "src/training/run_feature_pipeline.py"
    with open(runner_path, 'w') as f:
        f.write(runner_content)
    
    os.chmod(runner_path, 0o755)
    print(f"\n✅ Created pipeline runner script: {runner_path}")


def main():
    """Main function."""
    print("🔧 Updating pipeline for new Step 7 and Step 8...")
    
    # Update configuration
    update_training_config()
    
    # Provide Step 9 update instructions
    update_step09_for_new_features()
    
    # Create pipeline runner
    create_pipeline_runner()
    
    print("\n✅ Pipeline update complete!")
    print("\nNext steps:")
    print("1. Review and adjust the configuration in config/training_config.json")
    print("2. Update Step 9 to use Step 8 outputs as described above")
    print("3. Run the feature pipeline: python src/training/run_feature_pipeline.py BTCUSDT binance")


if __name__ == "__main__":
    main()