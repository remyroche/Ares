"""
NAS-Driven Clustering Module

This module provides Neural Architecture Search (NAS) driven clustering
for short-term trading regime detection (5-30m timeframe) with micro-regime detection.

Features:
- NAS-driven regime detection with 10-15 different states
- Short-term trading optimization (5-30m timeframe)
- Micro-regime detection for subtle market changes
- Full compatibility with existing pipeline
- Timestamped regime data for LM model training
- Enhanced feature extraction (excluding complex/irrelevant features)
"""

from .core.nas_clusterer import NASClusterer, NASClusteringResult
from .core.nas_config import NASConfig, NASClusteringConfig
from .core.micro_regime_detector import MicroRegimeDetector
from .core.nas_feature_extractor import NASFeatureExtractor
from .core.nas_regime_analyzer import NASRegimeAnalyzer

from .components.nas_clustering_component import NASClusteringComponent
from .components.nas_regime_handler import NASRegimeHandler
from .components.nas_output_formatter import NASOutputFormatter

from .utils.nas_metrics import NASMetrics, NASClusteringMetrics
from .utils.nas_visualizer import NASVisualizer
from .utils.nas_validator import NASValidator

from .integration.nas_orchestrator import NASOrchestrator
from .integration.nas_pipeline_integration import NASPipelineIntegration

__all__ = [
    # Core NAS clustering
    'NASClusterer',
    'NASClusteringResult',
    'NASConfig',
    'NASClusteringConfig',
    
    # Micro-regime detection
    'MicroRegimeDetector',
    
    # Feature extraction
    'NASFeatureExtractor',
    
    # Regime analysis
    'NASRegimeAnalyzer',
    
    # Components
    'NASClusteringComponent',
    'NASRegimeHandler',
    'NASOutputFormatter',
    
    # Utilities
    'NASMetrics',
    'NASClusteringMetrics',
    'NASVisualizer',
    'NASValidator',
    
    # Integration
    'NASOrchestrator',
    'NASPipelineIntegration'
]