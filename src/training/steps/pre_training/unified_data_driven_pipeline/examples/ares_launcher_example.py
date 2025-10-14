"""
Ares Launcher Integration Example for Unified Data-Driven Pipeline

This example demonstrates how to use the unified pipeline with ares_launcher
command line arguments for different intensity levels.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.pre_training.unified_data_driven_pipeline.ares_launcher_integration import (
    create_ares_launcher_integration, AresLauncherConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.refactored_pipeline import (
    create_refactored_pipeline
)


def example_command_line_usage():
    """Example of command line usage with ares_launcher."""
    print("=" * 80)
    print("ARES LAUNCHER COMMAND LINE USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n1. Basic usage with different intensity levels:")
    print("   ares_launcher --intensity full --output-dir results/full/")
    print("   ares_launcher --intensity blank --output-dir results/blank/")
    print("   ares_launcher --intensity light --output-dir results/light/")
    
    print("\n2. Custom configuration overrides:")
    print("   ares_launcher --intensity blank --max-features 30 --max-period 50")
    print("   ares_launcher --intensity light --cv-splits 3 --computation-time 300")
    print("   ares_launcher --intensity full --enable-gpu --log-level DEBUG")
    
    print("\n3. Advanced configuration:")
    print("   ares_launcher --intensity blank \\")
    print("     --max-features 25 \\")
    print("     --min-features 5 \\")
    print("     --max-period 63 \\")
    print("     --max-lookback 50 \\")
    print("     --max-interactions 30 \\")
    print("     --cv-splits 3 \\")
    print("     --computation-time 600 \\")
    print("     --output-dir results/custom/ \\")
    print("     --log-level INFO")
    
    print("\n4. Performance testing:")
    print("   ares_launcher --intensity light --log-level WARNING --no-save-results")
    print("   ares_launcher --intensity blank --log-level WARNING --no-save-results")
    print("   ares_launcher --intensity full --log-level WARNING --no-save-results")


def example_programmatic_usage():
    """Example of programmatic usage with ares_launcher integration."""
    print("\n" + "=" * 80)
    print("PROGRAMMATIC USAGE WITH ARES LAUNCHER INTEGRATION")
    print("=" * 80)
    
    # Create integration instance
    integration = create_ares_launcher_integration()
    
    # Example 1: Parse command line arguments
    print("\n1. Parsing command line arguments:")
    print("   python ares_launcher_cli.py --intensity blank --max-features 25")
    
    # Simulate parsing arguments
    test_args = ['--intensity', 'blank', '--max-features', '25', '--log-level', 'INFO']
    config = integration.parse_arguments(test_args)
    
    print(f"   Parsed intensity: {config.intensity}")
    print(f"   Custom overrides: {config.custom_overrides}")
    print(f"   Log level: {config.log_level}")
    
    # Example 2: Create pipeline programmatically
    print("\n2. Creating pipeline programmatically:")
    
    # Create custom configuration
    custom_config = AresLauncherConfig(
        intensity="blank",
        custom_overrides={
            'feature_selection.multi_objective.max_features': 30,
            'period_optimization.max_period': 50
        },
        log_level="INFO",
        output_dir="results/programmatic/",
        save_results=True
    )
    
    # Create pipeline
    pipeline = integration.create_pipeline(custom_config)
    print(f"   Pipeline created with intensity: {custom_config.intensity}")
    print(f"   Custom overrides: {custom_config.custom_overrides}")
    
    # Cleanup
    pipeline.cleanup()


def example_intensity_comparison():
    """Example comparing different intensity levels."""
    print("\n" + "=" * 80)
    print("INTENSITY LEVEL COMPARISON")
    print("=" * 80)
    
    integration = create_ares_launcher_integration()
    
    intensities = ["light", "blank", "full"]
    
    print("\nIntensity Level Characteristics:")
    print("-" * 50)
    
    for intensity in intensities:
        # Create configuration for each intensity
        config = AresLauncherConfig(intensity=intensity)
        
        # Create pipeline to get configuration details
        pipeline = integration.create_pipeline(config)
        pipeline_config = pipeline.config
        
        print(f"\n{intensity.upper()} INTENSITY:")
        print(f"  Max features: {pipeline_config.feature_selection.multi_objective.max_features}")
        print(f"  Max period: {pipeline_config.period_optimization.max_period}")
        print(f"  Max lookback: {pipeline_config.lookback_optimization.max_lookback}")
        print(f"  Max interactions: {pipeline_config.interaction_generation.max_interactions}")
        print(f"  CV splits: {pipeline_config.feature_selection.cv_config.n_splits}")
        print(f"  Computation time: {pipeline_config.feature_selection.max_computation_time}s")
        print(f"  GPU enabled: {pipeline_config.vectorization.enable_gpu}")
        print(f"  Parallel enabled: {pipeline_config.vectorization.enable_parallel}")
        
        # Cleanup
        pipeline.cleanup()


def example_help_output():
    """Example of help output from ares_launcher."""
    print("\n" + "=" * 80)
    print("ARES LAUNCHER HELP OUTPUT")
    print("=" * 80)
    
    integration = create_ares_launcher_integration()
    parser = integration.create_argument_parser()
    
    print("\nHelp output:")
    print("-" * 40)
    parser.print_help()


def example_configuration_validation():
    """Example of configuration validation."""
    print("\n" + "=" * 80)
    print("CONFIGURATION VALIDATION")
    print("=" * 80)
    
    integration = create_ares_launcher_integration()
    
    # Test valid configurations
    valid_configs = [
        AresLauncherConfig(intensity="full"),
        AresLauncherConfig(intensity="blank", custom_overrides={'max-features': 30}),
        AresLauncherConfig(intensity="light", log_level="DEBUG")
    ]
    
    print("\nValid configurations:")
    for i, config in enumerate(valid_configs, 1):
        print(f"  {i}. Intensity: {config.intensity}")
        print(f"     Overrides: {config.custom_overrides}")
        print(f"     Log level: {config.log_level}")
        
        # Validate by creating pipeline
        try:
            pipeline = integration.create_pipeline(config)
            print(f"     Status: ✅ Valid")
            pipeline.cleanup()
        except Exception as e:
            print(f"     Status: ❌ Invalid - {e}")
    
    # Test invalid configurations
    print("\nInvalid configurations (should fail gracefully):")
    invalid_configs = [
        AresLauncherConfig(intensity="invalid"),
        AresLauncherConfig(intensity="full", custom_overrides={'invalid_param': 'value'})
    ]
    
    for i, config in enumerate(invalid_configs, 1):
        print(f"  {i}. Intensity: {config.intensity}")
        print(f"     Overrides: {config.custom_overrides}")
        
        try:
            pipeline = integration.create_pipeline(config)
            print(f"     Status: ✅ Valid (unexpected)")
            pipeline.cleanup()
        except Exception as e:
            print(f"     Status: ❌ Invalid - {e}")


def main():
    """Run all ares_launcher examples."""
    print("ARES LAUNCHER INTEGRATION EXAMPLES")
    print("=" * 80)
    
    try:
        example_command_line_usage()
        example_programmatic_usage()
        example_intensity_comparison()
        example_help_output()
        example_configuration_validation()
        
        print("\n" + "=" * 80)
        print("ALL ARES LAUNCHER EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\nTo use with ares_launcher:")
        print("1. Run: python ares_launcher_cli.py --help")
        print("2. Run: python ares_launcher_cli.py --intensity blank --max-features 25")
        print("3. Run: python ares_launcher_cli.py --intensity light --output-dir results/")
        
    except Exception as e:
        print(f"\nERROR: Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()