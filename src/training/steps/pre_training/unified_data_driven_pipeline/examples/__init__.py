"""
Examples for the Unified Data-Driven Feature Pipeline

This module contains comprehensive examples demonstrating how to use the
unified data-driven feature pipeline for various machine learning tasks.

Available Examples:
- basic_pipeline_example.py: Basic usage of the consolidated pipeline
- advanced_feature_engineering.py: Advanced feature engineering techniques
- time_series_analysis.py: Time series specific analysis examples
- multi_objective_optimization.py: Multi-objective feature selection examples
- statistical_analysis_examples.py: Statistical analysis framework examples
- cross_validation_examples.py: Cross-validation and time series CV examples
- performance_optimization.py: M1 optimization and performance tuning examples
- custom_objective_functions.py: Creating custom objective functions
- feature_interaction_examples.py: Feature interaction and engineering examples
- pipeline_configuration.py: Advanced pipeline configuration examples

Usage:
    from src.training.steps.pre_training.unified_data_driven_pipeline.examples import basic_pipeline_example
    
    # Run basic pipeline example
    basic_pipeline_example.run_example()
"""

# Import example modules for easy access
try:
    from .basic_pipeline_example import run_basic_pipeline_example
    from .advanced_feature_engineering import run_advanced_feature_engineering_example
    from .time_series_analysis import run_time_series_analysis_example
    from .multi_objective_optimization import run_multi_objective_optimization_example
    from .statistical_analysis_examples import run_statistical_analysis_examples
    from .cross_validation_examples import run_cross_validation_examples
    from .performance_optimization import run_performance_optimization_example
    from .custom_objective_functions import run_custom_objective_functions_example
    from .feature_interaction_examples import run_feature_interaction_examples
    from .pipeline_configuration import run_pipeline_configuration_example
    
    __all__ = [
        'run_basic_pipeline_example',
        'run_advanced_feature_engineering_example',
        'run_time_series_analysis_example',
        'run_multi_objective_optimization_example',
        'run_statistical_analysis_examples',
        'run_cross_validation_examples',
        'run_performance_optimization_example',
        'run_custom_objective_functions_example',
        'run_feature_interaction_examples',
        'run_pipeline_configuration_example'
    ]
    
except ImportError as e:
    # Examples not yet implemented - this is expected during development
    __all__ = []