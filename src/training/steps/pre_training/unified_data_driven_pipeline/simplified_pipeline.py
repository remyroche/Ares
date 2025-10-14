"""
Simplified Unified Data-Driven Pipeline

This module provides a clean, simplified interface to the unified pipeline
that eliminates the massive consolidated_pipeline.py file.
"""

from typing import Dict, List, Optional, Any
import pandas as pd

from .core.unified_pipeline import (
    UnifiedDataDrivenPipeline,
    ConsolidatedPipelineResult,
    create_unified_pipeline,
    process_with_unified_pipeline
)
from .core.config import (
    UnifiedPipelineConfig,
    create_default_config,
    create_high_performance_config,
    create_memory_efficient_config,
    create_fast_config
)

# Re-export the main classes and functions
__all__ = [
    'UnifiedDataDrivenPipeline',
    'ConsolidatedPipelineResult', 
    'create_unified_pipeline',
    'process_with_unified_pipeline',
    'UnifiedPipelineConfig',
    'create_default_config',
    'create_high_performance_config',
    'create_memory_efficient_config',
    'create_fast_config'
]

# Version info
__version__ = "2.0.0"
__author__ = "Ares Trading System"