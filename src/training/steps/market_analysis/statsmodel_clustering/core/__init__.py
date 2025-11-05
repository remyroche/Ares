"""
Core Components for Statsmodels Clustering

This module provides core components for statsmodels regime switching models,
including the enhanced MarkovRegressionAdapter and related classes.

Key Components:
- MarkovRegressionAdapter: Enhanced wrapper around statsmodels MarkovRegression
- MarkovRegressionConfig: Configuration for the adapter
- MarkovRegressionResult: Result container for the adapter
- ParameterMapper: Maps Pyro parameters to statsmodels format
- MarkovRegressionDiagnostics: Advanced model diagnostics
"""

from .markov_regression_adapter import (
    MarkovRegressionAdapter,
    MarkovRegressionConfig,
    MarkovRegressionResult,
    ParameterMapper,
    MarkovRegressionDiagnostics,
    create_enhanced_markov_regression_adapter
)

from .base_data_downloader import (
    BaseDataDownloader,
    StandardDataDownloader,
    create_data_downloader,
    download_clustering_data
)

__all__ = [
    'MarkovRegressionAdapter',
    'MarkovRegressionConfig',
    'MarkovRegressionResult',
    'ParameterMapper',
    'MarkovRegressionDiagnostics',
    'create_enhanced_markov_regression_adapter',
    'BaseDataDownloader',
    'StandardDataDownloader',
    'create_data_downloader',
    'download_clustering_data'
]