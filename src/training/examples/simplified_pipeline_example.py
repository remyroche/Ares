"""Example: Using the Simplified Training Pipeline

This example demonstrates how to use the new simplified and refactored
training components to run a complete training pipeline.
"""
import asyncio
from pathlib import Path
import pandas as pd
from src.training.core.training_manager import create_training_manager
from src.utils.logger import system_logger

async def run_basic_training_example() -> None:
    """Run a basic training pipeline example."""
    logger = system_logger.getChild('TrainingExample')
    config = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1m', 'data_dir': 'data/training', 'lookback_years': 1, 'data_sources': ['binance'], 'feature_engineering': {'enable_wavelets': True, 'enable_multi_timeframe': True, 'timeframes': ['5m', '15m', '1h']}, 'model_config': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}, 'optimization': {'n_trials': 50, 'enable_early_stopping': True}}
    logger.info('🚀 Starting simplified training pipeline example')
    try:
        logger.info('📦 Creating training manager...')
        manager = await create_training_manager(config)
        logger.info('🔄 Executing full pipeline...')
        result = await manager.train(symbol='BTCUSDT', exchange='binance', start_step=None, end_step=None, force_rerun=False)
        if result['success']:
            logger.info('✅ Training completed successfully!')
            report = result['execution_report']
            logger.info(f'📊 Execution Summary:')
            logger.info(f"  - Steps executed: {len(report['steps_executed'])}")
            logger.info(f"  - Steps skipped: {len(report['steps_skipped'])}")
            logger.info(f"  - Steps failed: {len(report['steps_failed'])}")
            logger.info(f"  - Total duration: {report['total_duration']:.2f}s")
            status = await manager.get_status()
            logger.info(f'📈 Pipeline Status:')
            logger.info(f"  - Total steps: {status['total_steps']}")
            logger.info(f"  - Completed: {len(status['completed_steps'])}")
            logger.info(f"  - Pending: {len(status['pending_steps'])}")
        else:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.exception(f'💥 Example failed with error: {e}')
    finally:
        if 'manager' in locals():
            await manager.cleanup()
            logger.info('🧹 Cleaned up resources')

async def run_partial_pipeline_example() -> None:
    """Run a partial pipeline example (specific steps only)."""
    logger = system_logger.getChild('PartialPipelineExample')
    config = {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '5m', 'data_dir': 'data/training'}
    logger.info('🚀 Starting partial pipeline example')
    try:
        manager = await create_training_manager(config)
        logger.info('📊 Running data preparation steps only...')
        result = await manager.train(symbol='ETHUSDT', exchange='binance', start_step='01', end_step='02', force_rerun=False)
        if result['success']:
            logger.info('✅ Data preparation completed!')
            logger.info('🔄 Continuing with feature engineering...')
            result = await manager.train(symbol='ETHUSDT', exchange='binance', start_step='03', end_step='06', force_rerun=False)
            if result['success']:
                logger.info('✅ Feature engineering completed!')
    except Exception as e:
        logger.exception(f'💥 Partial pipeline failed: {e}')
    finally:
        if 'manager' in locals():
            await manager.cleanup()

async def run_custom_step_example() -> None:
    """Example of running with custom step configuration."""
    logger = system_logger.getChild('CustomStepExample')
    config = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '15m', 'step_params': {'01': {'force_download': True, 'lookback_years': 2}, '06': {'enable_wavelets': False, 'timeframes': ['30m', '1h', '4h']}, '09': {'model_type': 'lightgbm', 'n_estimators': 200}}}
    logger.info('🚀 Starting custom step configuration example')
    try:
        manager = await create_training_manager(config)
        result = await manager.train(symbol='BTCUSDT', exchange='binance', force_rerun=True)
        if result['success']:
            logger.info('✅ Custom pipeline completed!')
    except Exception as e:
        logger.exception(f'💥 Custom pipeline failed: {e}')
    finally:
        if 'manager' in locals():
            await manager.cleanup()

async def inspect_pipeline_state() -> None:
    """Example of inspecting pipeline state and outputs."""
    logger = system_logger.getChild('PipelineInspection')
    config = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1m', 'data_dir': 'data/training'}
    logger.info('🔍 Inspecting pipeline state')
    try:
        manager = await create_training_manager(config)
        status = await manager.get_status()
        logger.info(f'📊 Current Pipeline Status:')
        logger.info(f"Total steps: {status['total_steps']}")
        if status['completed_steps']:
            logger.info('✅ Completed steps:')
            for step in status['completed_steps']:
                logger.info(f'  - {step}')
        if status['pending_steps']:
            logger.info('⏳ Pending steps:')
            for step in status['pending_steps']:
                logger.info(f'  - {step}')
        if 'pipeline_state_keys' in status:
            logger.info('📦 Available outputs:')
            for key in status['pipeline_state_keys'][:10]:
                logger.info(f'  - {key}')
        features_path = Path(config['data_dir']) / f"{config['exchange']}_{config['symbol']}_{config['timeframe']}_features_train.parquet"
        if features_path.exists():
            features = pd.read_parquet(features_path)
            logger.info(f'📈 Feature data shape: {features.shape}')
            logger.info(f'📊 Feature columns sample: {list(features.columns[:10])}')
    except Exception as e:
        logger.exception(f'💥 Inspection failed: {e}')
    finally:
        if 'manager' in locals():
            await manager.cleanup()

def main() -> None:
    """Main entry point for examples."""
    import sys
    examples = {'basic': run_basic_training_example, 'partial': run_partial_pipeline_example, 'custom': run_custom_step_example, 'inspect': inspect_pipeline_state}
    example_name = sys.argv[1] if len(sys.argv) > 1 else 'basic'
    if example_name in examples:
        print(f'Running {example_name} example...')
        asyncio.run(examples[example_name]())
    else:
        print(f'Unknown example: {example_name}')
        print(f"Available examples: {', '.join(examples.keys())}")
        sys.exit(1)
if __name__ == '__main__':
    main()