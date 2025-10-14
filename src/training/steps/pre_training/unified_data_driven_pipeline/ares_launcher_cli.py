#!/usr/bin/env python3
"""
Ares Launcher CLI for Unified Data-Driven Pipeline

This script provides a command-line interface that can be used with ares_launcher
to configure and run the unified pipeline with different intensity levels.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.pre_training.unified_data_driven_pipeline.ares_launcher_integration import (
    create_ares_launcher_integration, AresLauncherConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.refactored_pipeline import (
    RefactoredUnifiedPipeline
)


def setup_logging(log_level: str) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ares_launcher_pipeline.log')
        ]
    )


def create_sample_data_for_demo() -> tuple:
    """Create sample data for demonstration purposes.
    
    Returns:
        Tuple of (data, targets) for pipeline processing
    """
    import numpy as np
    import pandas as pd
    
    # Create sample financial data
    np.random.seed(42)
    n_samples = 1000
    
    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.abs(np.random.randn(n_samples) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.abs(np.random.randn(n_samples) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Create targets (returns)
    targets = data['close'].pct_change().dropna()
    data = data.iloc[1:]  # Align with targets
    
    return data, targets


def run_pipeline_with_config(config: AresLauncherConfig, 
                           data=None, 
                           targets=None) -> Optional[dict]:
    """Run the pipeline with the given configuration.
    
    Args:
        config: AresLauncherConfig instance
        data: Optional input data (uses sample data if None)
        targets: Optional target data (uses sample data if None)
        
    Returns:
        Dictionary with pipeline results or None if failed
    """
    try:
        # Create integration instance
        integration = create_ares_launcher_integration()
        
        # Create pipeline
        pipeline = integration.create_pipeline(config)
        
        # Use sample data if none provided
        if data is None or targets is None:
            print("Using sample data for demonstration...")
            data, targets = create_sample_data_for_demo()
        
        # Process data
        print(f"\nProcessing data with {config.intensity} intensity...")
        print(f"Data shape: {data.shape}")
        print(f"Targets shape: {targets.shape}")
        
        result = pipeline.process(data, targets, timeframe="15m")
        
        # Print results
        print(f"\nPipeline Results:")
        print(f"  Selected features: {len(result.selected_features)}")
        print(f"  Processing time: {result.total_processing_time:.2f}s")
        print(f"  Memory usage: {result.memory_usage:.2f} MB")
        print(f"  Quality score: {result.quality_score:.3f}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Errors: {len(result.errors)}")
        
        # Save results if requested
        if config.save_results and config.output_dir:
            output_path = Path(config.output_dir)
            success = result.save_result(output_path)
            if success:
                print(f"\nResults saved to: {output_path}")
            else:
                print(f"\nFailed to save results to: {output_path}")
        
        # Cleanup
        pipeline.cleanup()
        
        return {
            'success': True,
            'selected_features': len(result.selected_features),
            'processing_time': result.total_processing_time,
            'memory_usage': result.memory_usage,
            'quality_score': result.quality_score,
            'warnings': len(result.warnings),
            'errors': len(result.errors)
        }
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main function for ares_launcher CLI."""
    try:
        # Create integration instance
        integration = create_ares_launcher_integration()
        
        # Parse command line arguments
        config = integration.parse_arguments()
        
        # Set up logging
        setup_logging(config.log_level)
        
        # Print configuration summary
        integration.print_configuration_summary(config)
        
        # Run pipeline
        result = run_pipeline_with_config(config)
        
        if result and result['success']:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"   Features selected: {result['selected_features']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Quality score: {result['quality_score']:.3f}")
        else:
            print(f"\n❌ Pipeline failed!")
            if result:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()