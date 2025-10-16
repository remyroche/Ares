"""
TPrint Configuration for NAS/TAS Pipeline

This module provides specialized tprint configurations optimized for the NAS/TAS
joint pipeline, ensuring comprehensive logging and monitoring throughout the
entire system.
"""

from src.utils.tprint import TPrintConfig, LogLevel, TimestampFormat
from typing import Optional, Dict, Any
from pathlib import Path
import os

class NASPipelineTPrintConfig(TPrintConfig):
    """Specialized tprint configuration for NAS pipeline."""

    def __init__(self, output_directory: str = "nas_tas_output", **kwargs):
        super().__init__(
            timestamp_format=TimestampFormat.WITH_MICROSECONDS,
            use_colors=True,
            output_to_console=True,
            output_to_file=True,
            output_file=f"{output_directory}/nas_pipeline.log",
            min_log_level=LogLevel.INFO,
            enable_structured_logging=True,
            integrate_with_logging=True,
            auto_log_prints=True,
            capture_print_to_tprint=True,
            single_file_per_run=True,
            **kwargs
        )

        # NAS-specific settings
        self.pipeline_type = "NAS"
        self.enable_architecture_logging = True
        self.enable_training_logging = True
        self.enable_evaluation_logging = True

class TASPipelineTPrintConfig(TPrintConfig):
    """Specialized tprint configuration for TAS pipeline."""

    def __init__(self, output_directory: str = "nas_tas_output", **kwargs):
        super().__init__(
            timestamp_format=TimestampFormat.WITH_MICROSECONDS,
            use_colors=True,
            output_to_console=True,
            output_to_file=True,
            output_file=f"{output_directory}/tas_pipeline.log",
            min_log_level=LogLevel.INFO,
            enable_structured_logging=True,
            integrate_with_logging=True,
            auto_log_prints=True,
            capture_print_to_tprint=True,
            single_file_per_run=True,
            **kwargs
        )

        # TAS-specific settings
        self.pipeline_type = "TAS"
        self.enable_tree_logging = True
        self.enable_ensemble_logging = True
        self.enable_optimization_logging = True

class HybridPipelineTPrintConfig(TPrintConfig):
    """Specialized tprint configuration for Hybrid NAS/TAS pipeline."""

    def __init__(self, output_directory: str = "nas_tas_output", **kwargs):
        super().__init__(
            timestamp_format=TimestampFormat.WITH_MICROSECONDS,
            use_colors=True,
            output_to_console=True,
            output_to_file=True,
            output_file=f"{output_directory}/hybrid_pipeline.log",
            min_log_level=LogLevel.INFO,
            enable_structured_logging=True,
            integrate_with_logging=True,
            auto_log_prints=True,
            capture_print_to_tprint=True,
            single_file_per_run=True,
            **kwargs
        )

        # Hybrid-specific settings
        self.pipeline_type = "HYBRID"
        self.enable_architecture_logging = True
        self.enable_training_logging = True
        self.enable_evaluation_logging = True
        self.enable_tree_logging = True
        self.enable_ensemble_logging = True
        self.enable_optimization_logging = True
        self.enable_hybrid_logging = True

class UnifiedPipelineTPrintConfig(TPrintConfig):
    """Comprehensive tprint configuration for unified NAS/TAS pipeline."""

    def __init__(self, output_directory: str = "nas_tas_output", **kwargs):
        super().__init__(
            timestamp_format=TimestampFormat.WITH_MICROSECONDS,
            use_colors=True,
            output_to_console=True,
            output_to_file=True,
            output_file=f"{output_directory}/unified_pipeline.log",
            min_log_level=LogLevel.INFO,
            enable_structured_logging=True,
            integrate_with_logging=True,
            auto_log_prints=True,
            capture_print_to_tprint=True,
            single_file_per_run=True,
            **kwargs
        )

        # Unified pipeline settings
        self.pipeline_type = "UNIFIED"
        self.enable_comprehensive_logging = True
        self.enable_performance_monitoring = True
        self.enable_error_tracking = True
        self.enable_architecture_logging = True
        self.enable_training_logging = True
        self.enable_evaluation_logging = True
        self.enable_tree_logging = True
        self.enable_ensemble_logging = True
        self.enable_optimization_logging = True
        self.enable_hybrid_logging = True
        self.enable_data_processing_logging = True
        self.enable_result_management_logging = True

def create_nas_tprint_config(output_directory: str = "nas_tas_output") -> NASPipelineTPrintConfig:
    """Create NAS-specific tprint configuration."""
    return NASPipelineTPrintConfig(output_directory)

def create_tas_tprint_config(output_directory: str = "nas_tas_output") -> TASPipelineTPrintConfig:
    """Create TAS-specific tprint configuration."""
    return TASPipelineTPrintConfig(output_directory)

def create_hybrid_tprint_config(output_directory: str = "nas_tas_output") -> HybridPipelineTPrintConfig:
    """Create Hybrid NAS/TAS tprint configuration."""
    return HybridPipelineTPrintConfig(output_directory)

def create_unified_tprint_config(output_directory: str = "nas_tas_output") -> UnifiedPipelineTPrintConfig:
    """Create unified NAS/TAS tprint configuration."""
    return UnifiedPipelineTPrintConfig(output_directory)

def get_pipeline_tprint_config(
    pipeline_type: str,
    output_directory: str = "nas_tas_output"
) -> TPrintConfig:
    """Get appropriate tprint configuration for pipeline type."""

    config_map = {
        "NAS": create_nas_tprint_config,
        "TAS": create_tas_tprint_config,
        "HYBRID": create_hybrid_tprint_config,
        "UNIFIED": create_unified_tprint_config
    }

    if pipeline_type.upper() not in config_map:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. "
                        f"Supported types: {list(config_map.keys())}")

    return config_map[pipeline_type.upper()](output_directory)

def configure_pipeline_logging(
    pipeline_type: str,
    output_directory: str = "nas_tas_output",
    log_level: LogLevel = LogLevel.INFO,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> TPrintConfig:
    """
    Configure comprehensive logging for NAS/TAS pipeline.

    Args:
        pipeline_type: Type of pipeline (NAS, TAS, HYBRID, UNIFIED)
        output_directory: Directory for log files
        log_level: Minimum log level
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging

    Returns:
        Configured TPrintConfig instance
    """
    from src.utils.tprint import configure_tprint

    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Get pipeline-specific configuration
    config = get_pipeline_tprint_config(pipeline_type, output_directory)

    # Override settings
    config.min_log_level = log_level
    config.output_to_console = enable_console_logging
    config.output_to_file = enable_file_logging

    # Apply configuration
    configure_tprint(config)

    return config

def setup_comprehensive_logging(
    pipeline_type: str = "UNIFIED",
    output_directory: str = "nas_tas_output",
    log_level: LogLevel = LogLevel.INFO
) -> Dict[str, Any]:
    """
    Setup comprehensive logging for NAS/TAS pipeline.

    Args:
        pipeline_type: Type of pipeline
        output_directory: Directory for log files
        log_level: Minimum log level

    Returns:
        Dictionary with logging configuration details
    """
    from src.utils.tprint import configure_tprint, get_tprint_config

    # Configure logging
    config = configure_pipeline_logging(
        pipeline_type=pipeline_type,
        output_directory=output_directory,
        log_level=log_level
    )

    # Get configuration details
    config_details = {
        "pipeline_type": pipeline_type,
        "output_directory": output_directory,
        "log_level": log_level.value,
        "config": get_tprint_config().to_dict() if hasattr(get_tprint_config(), 'to_dict') else {},
        "log_file": str(config.output_file) if config.output_file else None,
        "console_logging": config.output_to_console,
        "file_logging": config.output_to_file,
        "structured_logging": config.enable_structured_logging,
        "auto_log_prints": config.auto_log_prints
    }

    return config_details

# Convenience functions for quick setup
def setup_nas_logging(output_directory: str = "nas_tas_output") -> Dict[str, Any]:
    """Setup logging for NAS pipeline."""
    return setup_comprehensive_logging("NAS", output_directory)

def setup_tas_logging(output_directory: str = "nas_tas_output") -> Dict[str, Any]:
    """Setup logging for TAS pipeline."""
    return setup_comprehensive_logging("TAS", output_directory)

def setup_hybrid_logging(output_directory: str = "nas_tas_output") -> Dict[str, Any]:
    """Setup logging for Hybrid pipeline."""
    return setup_comprehensive_logging("HYBRID", output_directory)

def setup_unified_logging(output_directory: str = "nas_tas_output") -> Dict[str, Any]:
    """Setup logging for Unified pipeline."""
    return setup_comprehensive_logging("UNIFIED", output_directory)
