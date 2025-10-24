#!/usr/bin/env python3
"""
CLI Interface for ML Model Trainer

This script provides a command-line interface for the unified ML model trainer pipeline.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.ml_model_trainer import MLModelTrainer, MLModelTrainerConfig, ModelType
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_success


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified ML Model Trainer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default configs
  python cli_ml_model_trainer.py --timeframe 15m
  
  # Train specific model types
  python cli_ml_model_trainer.py --model-types analyst_base tactician_base
  
  # Use custom config directory
  python cli_ml_model_trainer.py --config-dir custom_configs/
  
  # Enable parallel training
  python cli_ml_model_trainer.py --parallel --max-workers 8
  
  # Verbose output
  python cli_ml_model_trainer.py --verbose
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model-types',
        nargs='+',
        choices=['analyst_base', 'analyst_ensemble', 'tactician_base', 'tactician_ensemble'],
        default=['analyst_base', 'analyst_ensemble', 'tactician_base', 'tactician_ensemble'],
        help='Model types to train (default: all)'
    )
    
    # Configuration
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config/ml_model_trainer/',
        help='Directory containing configuration files (default: config/ml_model_trainer/)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Timeframe for training (default: 15m)'
    )
    
    # Training options
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    # Performance options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel training'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/ml_model_trainer',
        help='Output directory for results (default: results/ml_model_trainer)'
    )
    
    parser.add_argument(
        '--no-save-models',
        action='store_true',
        help='Do not save trained models'
    )
    
    parser.add_argument(
        '--no-save-predictions',
        action='store_true',
        help='Do not save predictions'
    )
    
    parser.add_argument(
        '--no-save-reports',
        action='store_true',
        help='Do not generate reports'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Data options
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to input data file (CSV, Parquet, or JSON)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def validate_config_files(config_dir: str, model_types: List[str]) -> Dict[ModelType, str]:
    """Validate that all required configuration files exist."""
    config_paths = {}
    config_dir_path = Path(config_dir)
    
    if not config_dir_path.exists():
        tprint_error(f"Configuration directory does not exist: {config_dir}")
        sys.exit(1)
    
    for model_type_str in model_types:
        model_type = ModelType(model_type_str)
        config_file = config_dir_path / f"{model_type_str}_config.yaml"
        
        if not config_file.exists():
            tprint_error(f"Configuration file not found: {config_file}")
            sys.exit(1)
        
        config_paths[model_type] = str(config_file)
        tprint_info(f"✅ Found config: {config_file}")
    
    return config_paths


def load_data(data_file: str = None):
    """Load input data."""
    if data_file:
        data_path = Path(data_file)
        if not data_path.exists():
            tprint_error(f"Data file not found: {data_file}")
            sys.exit(1)
        
        # Load data based on file extension
        if data_path.suffix == '.csv':
            import pandas as pd
            data = pd.read_csv(data_path)
        elif data_path.suffix == '.parquet':
            import pandas as pd
            data = pd.read_parquet(data_path)
        elif data_path.suffix == '.json':
            import json
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            tprint_error(f"Unsupported data file format: {data_path.suffix}")
            sys.exit(1)
        
        tprint_info(f"📊 Loaded data from {data_file}")
        return data
    else:
        # Return placeholder data for demonstration
        import numpy as np
        tprint_info("📊 Using placeholder data for demonstration")
        return {
            'features': np.random.randn(1000, 50),
            'targets': np.random.randint(0, 2, 1000),
            'metadata': {'timeframe': '15m', 'n_samples': 1000}
        }


async def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.verbose:
        tprint_info("🔧 Verbose mode enabled")
    
    # Validate configuration files
    config_paths = validate_config_files(args.config_dir, args.model_types)
    
    # Load data
    data = load_data(args.data_file)
    
    # Create configuration
    config = MLModelTrainerConfig(
        model_types=[ModelType(mt) for mt in args.model_types],
        timeframe=args.timeframe,
        random_state=args.random_state,
        validation_split=args.validation_split,
        test_split=args.test_split,
        cv_folds=args.cv_folds,
        enable_parallel_training=args.parallel,
        max_workers=args.max_workers,
        enable_gpu=args.gpu,
        output_dir=args.output_dir,
        save_models=not args.no_save_models,
        save_predictions=not args.no_save_predictions,
        save_reports=not args.no_save_reports,
        enable_monitoring=True,
        log_level=args.log_level,
        verbose=args.verbose
    )
    
    # Create trainer
    trainer = MLModelTrainer(config, system_logger)
    
    # Print configuration summary
    tprint_info("🔧 Configuration Summary:")
    tprint_info(f"  Model Types: {[mt.value for mt in config.model_types]}")
    tprint_info(f"  Timeframe: {config.timeframe}")
    tprint_info(f"  Parallel Training: {config.enable_parallel_training}")
    tprint_info(f"  Max Workers: {config.max_workers}")
    tprint_info(f"  Output Directory: {config.output_dir}")
    tprint_info(f"  Random State: {config.random_state}")
    
    # Train models
    try:
        tprint_info("🚀 Starting ML model training pipeline")
        results = await trainer.train_models(data, config_paths)
        
        # Print results summary
        tprint_success("✅ Training completed successfully!")
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
        
        if successful_models < total_models:
            tprint_warning("Some models failed to train. Check logs for details.")
            sys.exit(1)
        
    except Exception as e:
        tprint_error(f"❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())