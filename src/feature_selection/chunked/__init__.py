"""
Feature Selection Chunked Processing Module

This module provides chunked processing capabilities for large datasets
with memory-efficient algorithms and hardware optimization.
"""

from .chunked_processor import (
    ChunkedFeatureProcessor,
    AdaptiveChunkProcessor,
    create_chunked_processor
)

__all__ = [
    'ChunkedFeatureProcessor',
    'AdaptiveChunkProcessor',
    'create_chunked_processor'
]
