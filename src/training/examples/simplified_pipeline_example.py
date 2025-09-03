"""Example: Using the Simplified Training Pipeline

This example demonstrates how to use the new simplified and refactored
training components to run a complete training pipeline.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Import the simplified training manager
from src.training.core.training_manager import create_training_manager
from src.utils.logger import system_logger


async def run_basic_training_example():
    """Run a basic training pipeline example."""
    logger = system_logger.getChild("TrainingExample")
    
    # 1. Configure the training pipeline
    config = {
        # Basic configuration
        "symbol": "BTCUSDT",
        "exchange": "binance", 
        "timeframe": "1m",
        "data_dir": "data/training",
        
        # Data collection settings
        "lookback_years": 1,
        "data_sources": ["binance"],
        
        # Feature engineering settings
        "feature_engineering": {
            "enable_wavelets": True,
            "enable_multi_timeframe": True,
            "timeframes": ["5m", "15m", "1h"]
        },
        
        # Model training settings
        "model_config": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5
        },
        
        # Optimization settings
        "optimization": {
            "n_trials": 50,
            "enable_early_stopping": True
        }
    }
    
    logger.info("🚀 Starting simplified training pipeline example")
    
    try:
        # 2. Create and initialize the training manager
        logger.info("📦 Creating training manager...")
        manager = await create_training_manager(config)
        
        # 3. Run the complete pipeline
        logger.info("🔄 Executing full pipeline...")
        result = await manager.train(
            symbol="BTCUSDT",
            exchange="binance",
            start_step=None,  # Start from beginning
            end_step=None,    # Run to completion
            force_rerun=False # Use cached results where available
        )
        
        # 4. Check results
        if result["success"]:
            logger.info("✅ Training completed successfully!")
            
            # Print execution summary
            report = result["execution_report"]
            logger.info(f"📊 Execution Summary:")
            logger.info(f"  - Steps executed: {len(report['steps_executed'])}")
            logger.info(f"  - Steps skipped: {len(report['steps_skipped'])}")
            logger.info(f"  - Steps failed: {len(report['steps_failed'])}")
            logger.info(f"  - Total duration: {report['total_duration']:.2f}s")
            
            # Get pipeline status
            status = await manager.get_status()
            logger.info(f"📈 Pipeline Status:")
            logger.info(f"  - Total steps: {status['total_steps']}")
            logger.info(f"  - Completed: {len(status['completed_steps'])}")
            logger.info(f"  - Pending: {len(status['pending_steps'])}")
            
        else:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.exception(f"💥 Example failed with error: {e}")
    
    finally:
        # 5. Cleanup
        if 'manager' in locals():
            await manager.cleanup()
            logger.info("🧹 Cleaned up resources")


async def run_partial_pipeline_example():
    """Run a partial pipeline example (specific steps only)."""
    logger = system_logger.getChild("PartialPipelineExample")
    
    config = {
        "symbol": "ETHUSDT",
        "exchange": "binance",
        "timeframe": "5m",
        "data_dir": "data/training"
    }
    
    logger.info("🚀 Starting partial pipeline example")
    
    try:
        # Create training manager
        manager = await create_training_manager(config)
        
        # Run only data preparation steps (1-2)
        logger.info("📊 Running data preparation steps only...")
        result = await manager.train(
            symbol="ETHUSDT",
            exchange="binance",
            start_step="01",  # Start from step 1
            end_step="02",    # End at step 2
            force_rerun=False
        )
        
        if result["success"]:
            logger.info("✅ Data preparation completed!")
            
            # Later, continue from where we left off
            logger.info("🔄 Continuing with feature engineering...")
            result = await manager.train(
                symbol="ETHUSDT",
                exchange="binance",
                start_step="03",  # Continue from step 3
                end_step="06",    # Stop after features
                force_rerun=False
            )
            
            if result["success"]:
                logger.info("✅ Feature engineering completed!")
        
    except Exception as e:
        logger.exception(f"💥 Partial pipeline failed: {e}")
    
    finally:
        if 'manager' in locals():
            await manager.cleanup()


async def run_custom_step_example():
    """Example of running with custom step configuration."""
    logger = system_logger.getChild("CustomStepExample")
    
    # Custom configuration with step-specific parameters
    config = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "15m",
        
        # Step-specific parameters
        "step_params": {
            "01": {
                "force_download": True,
                "lookback_years": 2
            },
            "06": {
                "enable_wavelets": False,  # Disable wavelets for speed
                "timeframes": ["30m", "1h", "4h"]  # Custom timeframes
            },
            "09": {
                "model_type": "lightgbm",
                "n_estimators": 200
            }
        }
    }
    
    logger.info("🚀 Starting custom step configuration example")
    
    try:
        manager = await create_training_manager(config)
        
        # Run with custom configuration
        result = await manager.train(
            symbol="BTCUSDT",
            exchange="binance",
            force_rerun=True  # Force re-run to use new params
        )
        
        if result["success"]:
            logger.info("✅ Custom pipeline completed!")
            
    except Exception as e:
        logger.exception(f"💥 Custom pipeline failed: {e}")
    
    finally:
        if 'manager' in locals():
            await manager.cleanup()


async def inspect_pipeline_state():
    """Example of inspecting pipeline state and outputs."""
    logger = system_logger.getChild("PipelineInspection")
    
    config = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "data_dir": "data/training"
    }
    
    logger.info("🔍 Inspecting pipeline state")
    
    try:
        manager = await create_training_manager(config)
        
        # Get current status
        status = await manager.get_status()
        
        logger.info(f"📊 Current Pipeline Status:")
        logger.info(f"Total steps: {status['total_steps']}")
        
        # List completed steps
        if status['completed_steps']:
            logger.info("✅ Completed steps:")
            for step in status['completed_steps']:
                logger.info(f"  - {step}")
        
        # List pending steps
        if status['pending_steps']:
            logger.info("⏳ Pending steps:")
            for step in status['pending_steps']:
                logger.info(f"  - {step}")
        
        # Check specific outputs
        if 'pipeline_state_keys' in status:
            logger.info("📦 Available outputs:")
            for key in status['pipeline_state_keys'][:10]:  # First 10
                logger.info(f"  - {key}")
        
        # Example: Load and inspect feature data if available
        features_path = Path(config["data_dir"]) / f"{config['exchange']}_{config['symbol']}_{config['timeframe']}_features_train.parquet"
        
        if features_path.exists():
            features = pd.read_parquet(features_path)
            logger.info(f"📈 Feature data shape: {features.shape}")
            logger.info(f"📊 Feature columns sample: {list(features.columns[:10])}")
            
    except Exception as e:
        logger.exception(f"💥 Inspection failed: {e}")
    
    finally:
        if 'manager' in locals():
            await manager.cleanup()


def main():
    """Main entry point for examples."""
    import sys
    
    examples = {
        "basic": run_basic_training_example,
        "partial": run_partial_pipeline_example,
        "custom": run_custom_step_example,
        "inspect": inspect_pipeline_state
    }
    
    # Get example to run from command line
    example_name = sys.argv[1] if len(sys.argv) > 1 else "basic"
    
    if example_name in examples:
        print(f"Running {example_name} example...")
        asyncio.run(examples[example_name]())
    else:
        print(f"Unknown example: {example_name}")
        print(f"Available examples: {', '.join(examples.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage:
    # python simplified_pipeline_example.py basic
    # python simplified_pipeline_example.py partial
    # python simplified_pipeline_example.py custom
    # python simplified_pipeline_example.py inspect
    main()