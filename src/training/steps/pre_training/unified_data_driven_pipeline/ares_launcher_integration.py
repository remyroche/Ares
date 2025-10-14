"""
Ares Launcher Integration for Unified Data-Driven Pipeline

This module provides integration with the ares_launcher command system
to allow intensity configuration via command line arguments.
"""

import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from .core.simplified_config import (
    create_full_config, create_blank_config, create_light_config,
    create_config_by_intensity, list_available_intensities,
    PipelineIntensity
)
from .refactored_pipeline import (
    RefactoredUnifiedPipeline, create_refactored_pipeline
)


@dataclass
class AresLauncherConfig:
    """Configuration for ares_launcher integration."""
    
    intensity: str = "full"
    custom_overrides: Dict[str, Any] = None
    log_level: str = "INFO"
    output_dir: Optional[str] = None
    save_results: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.custom_overrides is None:
            self.custom_overrides = {}


class AresLauncherIntegration:
    """Integration class for ares_launcher command system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the ares_launcher integration.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.available_intensities = list_available_intensities()
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for ares_launcher integration.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description="Unified Data-Driven Pipeline with intensity configuration",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
Available intensity levels:
{self._format_intensity_help()}

Examples:
  ares_launcher --intensity full --output-dir results/
  ares_launcher --intensity blank --max-features 30
  ares_launcher --intensity light --log-level DEBUG
            """
        )
        
        # Intensity configuration
        parser.add_argument(
            '--intensity',
            type=str,
            choices=list(self.available_intensities.keys()),
            default='full',
            help='Pipeline intensity level (default: full)'
        )
        
        # Custom configuration overrides
        parser.add_argument(
            '--max-features',
            type=int,
            help='Maximum number of features to select'
        )
        
        parser.add_argument(
            '--min-features',
            type=int,
            help='Minimum number of features to select'
        )
        
        parser.add_argument(
            '--max-period',
            type=int,
            help='Maximum period for optimization'
        )
        
        parser.add_argument(
            '--max-lookback',
            type=int,
            help='Maximum lookback for optimization'
        )
        
        parser.add_argument(
            '--max-interactions',
            type=int,
            help='Maximum number of interactions to generate'
        )
        
        parser.add_argument(
            '--cv-splits',
            type=int,
            help='Number of cross-validation splits'
        )
        
        parser.add_argument(
            '--computation-time',
            type=float,
            help='Maximum computation time in seconds'
        )
        
        # Output configuration
        parser.add_argument(
            '--output-dir',
            type=str,
            help='Output directory for results'
        )
        
        parser.add_argument(
            '--save-results',
            action='store_true',
            default=True,
            help='Save results to output directory'
        )
        
        parser.add_argument(
            '--no-save-results',
            action='store_false',
            dest='save_results',
            help='Do not save results to output directory'
        )
        
        # Logging configuration
        parser.add_argument(
            '--log-level',
            type=str,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        
        # Pipeline configuration
        parser.add_argument(
            '--timeframe',
            type=str,
            default='15m',
            help='Target timeframe (default: 15m)'
        )
        
        parser.add_argument(
            '--enable-gpu',
            action='store_true',
            help='Enable GPU acceleration'
        )
        
        parser.add_argument(
            '--enable-parallel',
            action='store_true',
            default=True,
            help='Enable parallel processing (default: True)'
        )
        
        parser.add_argument(
            '--disable-parallel',
            action='store_false',
            dest='enable_parallel',
            help='Disable parallel processing'
        )
        
        return parser
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> AresLauncherConfig:
        """Parse command line arguments.
        
        Args:
            args: Optional list of arguments (defaults to sys.argv)
            
        Returns:
            AresLauncherConfig with parsed arguments
        """
        parser = self.create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Build custom overrides from parsed arguments
        custom_overrides = {}
        
        if parsed_args.max_features is not None:
            custom_overrides['feature_selection.multi_objective.max_features'] = parsed_args.max_features
        
        if parsed_args.min_features is not None:
            custom_overrides['feature_selection.multi_objective.min_features'] = parsed_args.min_features
        
        if parsed_args.max_period is not None:
            custom_overrides['period_optimization.max_period'] = parsed_args.max_period
        
        if parsed_args.max_lookback is not None:
            custom_overrides['lookback_optimization.max_lookback'] = parsed_args.max_lookback
        
        if parsed_args.max_interactions is not None:
            custom_overrides['interaction_generation.max_interactions'] = parsed_args.max_interactions
        
        if parsed_args.cv_splits is not None:
            custom_overrides['feature_selection.cv_config.n_splits'] = parsed_args.cv_splits
        
        if parsed_args.computation_time is not None:
            custom_overrides['feature_selection.max_computation_time'] = parsed_args.computation_time
            custom_overrides['period_optimization.max_computation_time'] = parsed_args.computation_time
            custom_overrides['lookback_optimization.max_computation_time'] = parsed_args.computation_time
        
        if parsed_args.enable_gpu:
            custom_overrides['vectorization.enable_gpu'] = True
        
        if not parsed_args.enable_parallel:
            custom_overrides['vectorization.enable_parallel'] = False
            custom_overrides['period_optimization.enable_parallel'] = False
        
        return AresLauncherConfig(
            intensity=parsed_args.intensity,
            custom_overrides=custom_overrides,
            log_level=parsed_args.log_level,
            output_dir=parsed_args.output_dir,
            save_results=parsed_args.save_results
        )
    
    def create_pipeline(self, config: AresLauncherConfig) -> RefactoredUnifiedPipeline:
        """Create a pipeline instance from ares_launcher configuration.
        
        Args:
            config: AresLauncherConfig instance
            
        Returns:
            Configured RefactoredUnifiedPipeline instance
        """
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create pipeline with intensity and custom overrides
        pipeline = create_refactored_pipeline(
            intensity=config.intensity,
            custom_overrides=config.custom_overrides,
            logger=self.logger
        )
        
        return pipeline
    
    def _format_intensity_help(self) -> str:
        """Format intensity help text.
        
        Returns:
            Formatted help text for intensity levels
        """
        help_lines = []
        for intensity, description in self.available_intensities.items():
            help_lines.append(f"  {intensity:<8} - {description}")
        return "\n".join(help_lines)
    
    def print_configuration_summary(self, config: AresLauncherConfig) -> None:
        """Print a summary of the configuration.
        
        Args:
            config: AresLauncherConfig to summarize
        """
        print("=" * 60)
        print("ARES LAUNCHER PIPELINE CONFIGURATION")
        print("=" * 60)
        print(f"Intensity: {config.intensity}")
        print(f"Description: {self.available_intensities[config.intensity]}")
        print(f"Log Level: {config.log_level}")
        print(f"Output Directory: {config.output_dir or 'Not specified'}")
        print(f"Save Results: {config.save_results}")
        
        if config.custom_overrides:
            print("\nCustom Overrides:")
            for key, value in config.custom_overrides.items():
                print(f"  {key}: {value}")
        else:
            print("\nNo custom overrides")
        
        print("=" * 60)


def create_ares_launcher_integration(logger: Optional[logging.Logger] = None) -> AresLauncherIntegration:
    """Create an ares_launcher integration instance.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        AresLauncherIntegration instance
    """
    return AresLauncherIntegration(logger)


def main():
    """Main function for ares_launcher integration."""
    integration = create_ares_launcher_integration()
    
    # Parse arguments
    config = integration.parse_arguments()
    
    # Print configuration summary
    integration.print_configuration_summary(config)
    
    # Create pipeline
    pipeline = integration.create_pipeline(config)
    
    print(f"\nPipeline created successfully!")
    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Intensity: {config.intensity}")
    
    # Note: In a real implementation, you would process data here
    print("\nNote: This is a configuration demonstration.")
    print("In a real implementation, you would process data with the pipeline here.")
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()