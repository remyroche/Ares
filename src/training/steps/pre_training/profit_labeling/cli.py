#!/usr/bin/env python3
"""
Command Line Interface for Enhanced Profit Labeling System

This module provides a command-line interface for running the enhanced profit
labeling system with various configurations and options.

Usage:
    python cli.py --config basic
    python cli.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --max-features 500
    python cli.py --config advanced --optimize --gpu

Author: AI Assistant
Date: 2025-01-10
"""

import argparse
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system import (
    EnhancedProfitLabelingSystem, ProfitLabelingConfig
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning


def load_config_from_file(config_name: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / "config" / "profit_labeling_configs.yaml"
    
    if not config_file.exists():
        tprint_error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            configs = yaml.safe_load(f)
        
        if config_name not in configs:
            tprint_error(f"Configuration '{config_name}' not found in {config_file}")
            tprint_info(f"Available configurations: {list(configs.keys())}")
            sys.exit(1)
        
        return configs[config_name]
    
    except Exception as e:
        tprint_error(f"Error loading configuration: {e}")
        sys.exit(1)


def create_config_from_args(args) -> ProfitLabelingConfig:
    """Create configuration from command line arguments."""
    config_dict = {}
    
    # Load base configuration if specified
    if args.config:
        base_config = load_config_from_file(args.config)
        config_dict.update(base_config)
    
    # Override with command line arguments
    if args.symbols:
        config_dict['symbols'] = args.symbols
    if args.timeframes:
        config_dict['timeframes'] = args.timeframes
    if args.max_features:
        config_dict['max_features'] = args.max_features
    if args.feature_selection:
        config_dict['feature_selection_method'] = args.feature_selection
    if args.volatility_threshold:
        config_dict['volatility_threshold'] = args.volatility_threshold
    if args.start_date:
        config_dict['start_date'] = args.start_date
    if args.end_date:
        config_dict['end_date'] = args.end_date
    if args.trials:
        config_dict['n_trials'] = args.trials
    if args.jobs:
        config_dict['n_jobs'] = args.jobs
    
    # Boolean flags
    if args.optimize:
        config_dict['enable_bayesian_optimization'] = True
    if args.gpu:
        config_dict['enable_gpu'] = True
    if args.no_parallel:
        config_dict['enable_parallel'] = False
    if args.memory_efficient:
        config_dict['memory_efficient'] = True
    if args.noise_gating:
        config_dict['enable_noise_gating'] = True
    if args.leakage_detection:
        config_dict['enable_leakage_detection'] = True
    
    return ProfitLabelingConfig(**config_dict)


def print_config_summary(config: ProfitLabelingConfig):
    """Print configuration summary."""
    tprint_info("📋 Configuration Summary:")
    tprint_info(f"  Symbols: {config.symbols}")
    tprint_info(f"  Timeframes: {config.timeframes}")
    tprint_info(f"  Date Range: {config.start_date} to {config.end_date}")
    tprint_info(f"  Max Features: {config.max_features}")
    tprint_info(f"  Feature Selection: {config.feature_selection_method}")
    tprint_info(f"  Volatility Threshold: {config.volatility_threshold}")
    tprint_info(f"  Bayesian Optimization: {config.enable_bayesian_optimization}")
    tprint_info(f"  GPU Enabled: {config.enable_gpu}")
    tprint_info(f"  Parallel Processing: {config.enable_parallel}")
    tprint_info(f"  Memory Efficient: {config.memory_efficient}")


def save_results(results: dict, output_file: str):
    """Save results to file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        tprint_success(f"✅ Results saved to {output_file}")
    except Exception as e:
        tprint_error(f"❌ Error saving results: {e}")


def print_results_summary(results: dict):
    """Print results summary."""
    tprint_info("\n📊 Results Summary:")
    tprint_info(f"  Datasets Processed: {len(results.get('data', {}))}")
    tprint_info(f"  Features Generated: {sum(results.get('features', {}).values())}")
    tprint_info(f"  Labels Generated: {sum(results.get('labels', {}).values())}")
    tprint_info(f"  Features Selected: {sum(results.get('selected_features', {}).values())}")
    
    if results.get('optimization'):
        tprint_info(f"  Optimization Parameters: {len(results['optimization'])}")
    
    if results.get('evaluation'):
        tprint_info(f"  Evaluation Datasets: {len(results['evaluation'])}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Profit Labeling System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use basic configuration
  python cli.py --config basic
  
  # Use advanced configuration with optimization
  python cli.py --config advanced --optimize
  
  # Custom configuration
  python cli.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --max-features 500
  
  # High-frequency trading setup
  python cli.py --config high_frequency --gpu --optimize
  
  # Research setup with custom parameters
  python cli.py --config research --trials 500 --jobs 16 --gpu
        """
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        choices=['basic', 'advanced', 'high_frequency', 'research', 'conservative', 
                'aggressive', 'gpu_optimized', 'memory_efficient', 'multi_asset'],
        help='Use predefined configuration'
    )
    config_group.add_argument(
        '--symbols', '-s',
        nargs='+',
        help='Trading symbols (e.g., BTCUSDT ETHUSDT)'
    )
    config_group.add_argument(
        '--timeframes', '-t',
        nargs='+',
        help='Timeframes (e.g., 1h 4h 1d)'
    )
    config_group.add_argument(
        '--max-features', '-f',
        type=int,
        help='Maximum number of features'
    )
    config_group.add_argument(
        '--feature-selection',
        choices=['mrmr', 'lasso', 'rfe', 'ensemble'],
        help='Feature selection method'
    )
    config_group.add_argument(
        '--volatility-threshold',
        type=float,
        help='Volatility threshold for labeling'
    )
    config_group.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD)'
    )
    config_group.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD)'
    )
    
    # Optimization options
    opt_group = parser.add_argument_group('Optimization')
    opt_group.add_argument(
        '--optimize', '-o',
        action='store_true',
        help='Enable Bayesian optimization'
    )
    opt_group.add_argument(
        '--trials',
        type=int,
        help='Number of optimization trials'
    )
    opt_group.add_argument(
        '--jobs',
        type=int,
        help='Number of parallel jobs'
    )
    
    # Hardware options
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration'
    )
    hw_group.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    hw_group.add_argument(
        '--memory-efficient',
        action='store_true',
        help='Enable memory-efficient mode'
    )
    
    # Quality options
    qual_group = parser.add_argument_group('Quality')
    qual_group.add_argument(
        '--noise-gating',
        action='store_true',
        help='Enable noise gating'
    )
    qual_group.add_argument(
        '--leakage-detection',
        action='store_true',
        help='Enable leakage detection'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output', '-o',
        help='Output file for results'
    )
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging level
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.ERROR)
    
    try:
        # Create configuration
        tprint_info("🔧 Creating configuration...")
        config = create_config_from_args(args)
        
        if args.verbose:
            print_config_summary(config)
        
        # Initialize system
        tprint_info("🚀 Initializing Enhanced Profit Labeling System...")
        system = EnhancedProfitLabelingSystem(config)
        
        # Run pipeline
        tprint_info("🏃 Running profit labeling pipeline...")
        results = system.run_full_pipeline()
        
        # Print summary
        print_results_summary(results)
        
        # Save results
        if args.output:
            save_results(results, args.output)
        else:
            # Default output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"profit_labeling_results_{timestamp}.json"
            save_results(results, output_file)
        
        tprint_success("🎉 Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        tprint_warning("⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        tprint_error(f"❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()