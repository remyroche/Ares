#!/usr/bin/env python3
"""
Example Usage of ML Model Trainer

This script demonstrates how to use the unified ML model trainer pipeline.
"""

import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.steps.models_training.training.ml_model_trainer import MLModelTrainer, MLModelTrainerConfig, ModelType
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_warning,
    tprint_data_preview, tprint_data_format, LogLevel
)
from src.utils.common_operations import safe_dataframe_operation, safe_array_operation
from src.utils.common_utilities import validate_dataframe, validate_array
from src.utils.math_validation import safe_statistical_operation, safe_divide, safe_log


async def create_sample_data():
    """Create sample data for demonstration using safe operations."""
    tprint_info("📊 Creating sample data with safe operations")
    
    # Generate sample features using safe operations
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Price features with safe operations
    price_features = safe_statistical_operation(
        np.random.randn(n_samples, 10),
        lambda x: np.where(np.isfinite(x), x, 0.0)
    )
    
    # Volume features with safe operations
    volume_features = safe_statistical_operation(
        np.random.exponential(1.0, (n_samples, 10)),
        lambda x: np.where(np.isfinite(x), x, 1.0)
    )
    
    # Technical indicators with safe operations
    technical_features = safe_statistical_operation(
        np.random.randn(n_samples, 20),
        lambda x: np.where(np.isfinite(x), x, 0.0)
    )
    
    # Market microstructure with safe operations
    microstructure_features = safe_statistical_operation(
        np.random.randn(n_samples, 10),
        lambda x: np.where(np.isfinite(x), x, 0.0)
    )
    
    # Combine all features using safe operations
    features = safe_array_operation(
        np.array([]),
        lambda x: np.hstack([price_features, volume_features, technical_features, microstructure_features])
    )
    
    # Generate targets with safe operations
    # Analyst targets (binary classification)
    analyst_targets = safe_statistical_operation(
        np.random.randint(0, 2, n_samples),
        lambda x: np.where(np.isfinite(x), x, 0)
    )
    analyst_confidence = safe_statistical_operation(
        np.random.uniform(0, 1, n_samples),
        lambda x: np.where(np.isfinite(x), x, 0.5)
    )
    
    # Tactician targets (regression) with safe operations
    entry_timing = safe_statistical_operation(
        np.random.uniform(-1, 1, n_samples),
        lambda x: np.where(np.isfinite(x), x, 0.0)
    )
    exit_timing = safe_statistical_operation(
        np.random.uniform(-1, 1, n_samples),
        lambda x: np.where(np.isfinite(x), x, 0.0)
    )
    position_sizing = safe_statistical_operation(
        np.random.uniform(0, 1, n_samples),
        lambda x: np.where(np.isfinite(x), x, 0.5)
    )
    
    # Create timestamps
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    # Create DataFrame using safe operations
    data = safe_dataframe_operation(
        pd.DataFrame(),
        lambda df: pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    )
    
    # Add columns using safe operations
    data = safe_dataframe_operation(
        data,
        lambda df: df.assign(
            timestamp=timestamps,
            analyst_target=analyst_targets,
            analyst_confidence=analyst_confidence,
            entry_timing=entry_timing,
            exit_timing=exit_timing,
            position_sizing=position_sizing,
            regime=np.random.choice(['low_vol', 'normal', 'high_vol'], n_samples)
        )
    )
    
    # Validate the created data
    if not validate_dataframe(data):
        tprint_error("Invalid sample data created")
        raise ValueError("Invalid sample data created")
    
    tprint_data_preview(features, "Sample features")
    tprint_data_preview(analyst_targets, "Analyst targets")
    tprint_data_preview(entry_timing, "Entry timing targets")
    tprint_data_format(f"Sample data - Features: {features.shape}, Analyst targets: {analyst_targets.shape}, Tactician targets: {entry_timing.shape}", LogLevel.INFO)
    
    tprint_success(f"✅ Created sample data with {n_samples} samples and {n_features} features")
    
    return data


async def run_analyst_base_training():
    """Run analyst base model training."""
    tprint_info("🔍 Running Analyst Base Model Training")
    
    # Create configuration
    config = MLModelTrainerConfig(
        model_types=[ModelType.ANALYST_BASE],
        timeframe="15m",
        enable_parallel_training=False,  # Sequential for demo
        max_workers=1,
        output_dir="results/analyst_base_example",
        save_models=True,
        save_predictions=True,
        save_reports=True,
        verbose=True
    )
    
    # Create trainer
    trainer = MLModelTrainer(config, system_logger)
    
    # Create sample data
    data = await create_sample_data()
    
    # Prepare data for training
    data_dict = {
        'features': data.drop(['timestamp', 'analyst_target', 'analyst_confidence', 
                              'entry_timing', 'exit_timing', 'position_sizing', 'regime'], axis=1).values,
        'targets': data[['analyst_target', 'analyst_confidence']].values,
        'metadata': {
            'timeframe': '15m',
            'n_samples': len(data),
            'feature_names': [f'feature_{i}' for i in range(50)],
            'target_names': ['analyst_target', 'analyst_confidence']
        }
    }
    
    # Define config path
    config_paths = {
        ModelType.ANALYST_BASE: "src/training/steps/models_training/config/ml_model_trainer/analyst_base_config.yaml"
    }
    
    # Train models
    try:
        results = await trainer.train_models(data_dict, config_paths)
        
        # Print results
        for model_type, model_results in results.items():
            tprint_info(f"\n{model_type.value} Results:")
            for result in model_results:
                if result.success:
                    tprint_success(f"  ✅ {result.model_name}: {result.training_time:.2f}s")
                    tprint_info(f"    Metrics: {result.metrics}")
                else:
                    tprint_error(f"  ❌ {result.model_name}: {result.error_message}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Training failed: {e}")
        raise


async def run_tactician_base_training():
    """Run tactician base model training."""
    tprint_info("⚔️ Running Tactician Base Model Training")
    
    # Create configuration
    config = MLModelTrainerConfig(
        model_types=[ModelType.TACTICIAN_BASE],
        timeframe="15m",
        enable_parallel_training=False,  # Sequential for demo
        max_workers=1,
        output_dir="results/tactician_base_example",
        save_models=True,
        save_predictions=True,
        save_reports=True,
        verbose=True
    )
    
    # Create trainer
    trainer = MLModelTrainer(config, system_logger)
    
    # Create sample data
    data = await create_sample_data()
    
    # Prepare data for training
    data_dict = {
        'features': data.drop(['timestamp', 'analyst_target', 'analyst_confidence', 
                              'entry_timing', 'exit_timing', 'position_sizing', 'regime'], axis=1).values,
        'targets': data[['entry_timing', 'exit_timing', 'position_sizing']].values,
        'metadata': {
            'timeframe': '15m',
            'n_samples': len(data),
            'feature_names': [f'feature_{i}' for i in range(50)],
            'target_names': ['entry_timing', 'exit_timing', 'position_sizing']
        }
    }
    
    # Define config path
    config_paths = {
        ModelType.TACTICIAN_BASE: "src/training/steps/models_training/config/ml_model_trainer/tactician_base_config.yaml"
    }
    
    # Train models
    try:
        results = await trainer.train_models(data_dict, config_paths)
        
        # Print results
        for model_type, model_results in results.items():
            tprint_info(f"\n{model_type.value} Results:")
            for result in model_results:
                if result.success:
                    tprint_success(f"  ✅ {result.model_name}: {result.training_time:.2f}s")
                    tprint_info(f"    Metrics: {result.metrics}")
                else:
                    tprint_error(f"  ❌ {result.model_name}: {result.error_message}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Training failed: {e}")
        raise


async def run_full_pipeline():
    """Run the full ML model training pipeline."""
    tprint_info("🚀 Running Full ML Model Training Pipeline")
    
    # Create configuration for all model types
    config = MLModelTrainerConfig(
        model_types=[
            ModelType.ANALYST_BASE,
            ModelType.ANALYST_ENSEMBLE,
            ModelType.TACTICIAN_BASE,
            ModelType.TACTICIAN_ENSEMBLE
        ],
        timeframe="15m",
        enable_parallel_training=True,
        max_workers=4,
        output_dir="results/full_pipeline_example",
        save_models=True,
        save_predictions=True,
        save_reports=True,
        verbose=True
    )
    
    # Create trainer
    trainer = MLModelTrainer(config, system_logger)
    
    # Create sample data
    data = await create_sample_data()
    
    # Prepare data for training
    data_dict = {
        'features': data.drop(['timestamp', 'analyst_target', 'analyst_confidence', 
                              'entry_timing', 'exit_timing', 'position_sizing', 'regime'], axis=1).values,
        'targets': data[['analyst_target', 'analyst_confidence', 'entry_timing', 
                        'exit_timing', 'position_sizing']].values,
        'metadata': {
            'timeframe': '15m',
            'n_samples': len(data),
            'feature_names': [f'feature_{i}' for i in range(50)],
            'target_names': ['analyst_target', 'analyst_confidence', 'entry_timing', 
                           'exit_timing', 'position_sizing']
        }
    }
    
    # Define config paths
    config_paths = {
        ModelType.ANALYST_BASE: "src/training/steps/models_training/config/ml_model_trainer/analyst_base_config.yaml",
        ModelType.ANALYST_ENSEMBLE: "src/training/steps/models_training/config/ml_model_trainer/analyst_ensemble_config.yaml",
        ModelType.TACTICIAN_BASE: "src/training/steps/models_training/config/ml_model_trainer/tactician_base_config.yaml",
        ModelType.TACTICIAN_ENSEMBLE: "src/training/steps/models_training/config/ml_model_trainer/tactician_ensemble_config.yaml"
    }
    
    # Train models
    try:
        results = await trainer.train_models(data_dict, config_paths)
        
        # Print results summary
        tprint_success("✅ Full pipeline completed successfully!")
        tprint_info("📊 Results Summary:")
        
        total_models = 0
        successful_models = 0
        
        for model_type, model_results in results.items():
            tprint_info(f"\n{model_type.value}:")
            for result in model_results:
                total_models += 1
                if result.success:
                    successful_models += 1
                    tprint_success(f"  ✅ {result.model_name}: {result.training_time:.2f}s")
                    if result.metrics:
                        tprint_info(f"    Metrics: {result.metrics}")
                else:
                    tprint_error(f"  ❌ {result.model_name}: {result.error_message}")
        
        tprint_info(f"\nOverall: {successful_models}/{total_models} models trained successfully")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Full pipeline failed: {e}")
        raise


async def main():
    """Main function."""
    tprint_info("🎯 ML Model Trainer Example")
    tprint_info("=" * 50)
    
    try:
        # Example 1: Analyst Base Training
        tprint_info("\n📊 Example 1: Analyst Base Model Training")
        tprint_info("-" * 40)
        await run_analyst_base_training()
        
        # Example 2: Tactician Base Training
        tprint_info("\n⚔️ Example 2: Tactician Base Model Training")
        tprint_info("-" * 40)
        await run_tactician_base_training()
        
        # Example 3: Full Pipeline
        tprint_info("\n🚀 Example 3: Full ML Model Training Pipeline")
        tprint_info("-" * 40)
        await run_full_pipeline()
        
        tprint_success("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())