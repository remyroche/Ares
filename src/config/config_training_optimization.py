# src/config/config_training_optimization.py

"""
Configuration file for optimizable training optimization parameters from other steps.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class TrainingOptimizationConfig:
    """Optimizable training optimization parameters from other steps."""
    
    # Step 2: Market Regime Classification
    adx_trend_threshold: float = 25.0
    adx_sideways_threshold: float = 20.0
    ema_sep_min_ratio: float = 0.0
    max_calibration_iters: int = 6
    max_regime_dominance: float = 0.85
    min_regime_frequency: float = 0.03
    max_regime_switching: float = 0.6
    max_stuck_ratio: float = 0.4
    
    # Step 3: HMM Regime Discovery
    min_quality_score: float = 0.7
    max_correlation: float = 0.95
    progress_interval: int = 10
    
    # Step 4: Processing & Labeling
    completeness_threshold: float = 0.95
    min_data_points: int = 100
    min_labeled_rows: int = 1000
    min_label_balance: float = 0.05
    max_label_balance: float = 0.95
    splitting_time_minutes: float = 30.0
    labeling_time_minutes: float = 45.0
    
    # Step 5: HMM-Based Training
    learning_rate: float = 0.0001
    architecture_optimization_enabled: bool = False
    
    # Step 6: Analyst Enhancement
    stability_threshold: float = 0.7
    mi_threshold: float = 0.01
    feature_selection_threshold: float = 0.2
    
    # Model-specific hyperparameters
    # LightGBM
    lgb_learning_rate: float = 0.05
    lgb_max_depth: int = 6
    lgb_min_child_samples: int = 20
    lgb_num_leaves: int = 31
    
    # Neural Networks
    nn_learning_rate: float = 0.001
    nn_max_iter: int = 500
    nn_hidden_layer_sizes: tuple = (100, 50)
    
    # Random Forest
    rf_max_depth: int = 10
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_n_estimators: int = 100
    
    # Step 11: Confidence Calibration
    calibration_accuracy: float = 0.7
    calibration_time_minutes: float = 60.0
    
    # Performance thresholds
    model_performance_threshold: float = 0.7
    data_quality_threshold: float = 0.95
    artifact_completeness_threshold: float = 0.9
    
    # Memory and performance
    memory_threshold_gb: float = 8.0
    cpu_threshold_percent: float = 80.0
    disk_threshold_gb: float = 5.0
    monitor_interval: float = 30.0
    failure_threshold: int = 3


def get_training_optimization_config() -> TrainingOptimizationConfig:
    """Get training optimization configuration."""
    return TrainingOptimizationConfig()


def get_training_optimization_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for training optimization."""
    return {
        # Step 2: Market Regime Classification
        "adx_trend_threshold": {"min": 20.0, "max": 35.0, "type": "float"},
        "adx_sideways_threshold": {"min": 15.0, "max": 30.0, "type": "float"},
        "ema_sep_min_ratio": {"min": 0.0, "max": 0.1, "type": "float"},
        "max_calibration_iters": {"min": 3, "max": 10, "type": "int"},
        "max_regime_dominance": {"min": 0.8, "max": 0.95, "type": "float"},
        "min_regime_frequency": {"min": 0.02, "max": 0.08, "type": "float"},
        "max_regime_switching": {"min": 0.5, "max": 0.8, "type": "float"},
        "max_stuck_ratio": {"min": 0.3, "max": 0.6, "type": "float"},
        
        # Step 3: HMM Regime Discovery
        "min_quality_score": {"min": 0.6, "max": 0.9, "type": "float"},
        "max_correlation": {"min": 0.9, "max": 0.98, "type": "float"},
        "progress_interval": {"min": 5, "max": 20, "type": "int"},
        
        # Step 4: Processing & Labeling
        "completeness_threshold": {"min": 0.9, "max": 0.99, "type": "float"},
        "min_data_points": {"min": 50, "max": 200, "type": "int"},
        "min_labeled_rows": {"min": 500, "max": 2000, "type": "int"},
        "min_label_balance": {"min": 0.03, "max": 0.1, "type": "float"},
        "max_label_balance": {"min": 0.9, "max": 0.98, "type": "float"},
        "splitting_time_minutes": {"min": 20.0, "max": 60.0, "type": "float"},
        "labeling_time_minutes": {"min": 30.0, "max": 90.0, "type": "float"},
        
        # Step 5: HMM-Based Training
        "learning_rate": {"min": 0.00001, "max": 0.001, "type": "float"},
        
        # Step 6: Analyst Enhancement
        "stability_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "mi_threshold": {"min": 0.005, "max": 0.02, "type": "float"},
        "feature_selection_threshold": {"min": 0.1, "max": 0.4, "type": "float"},
        
        # Model-specific hyperparameters
        # LightGBM
        "lgb_learning_rate": {"min": 0.01, "max": 0.2, "type": "float"},
        "lgb_max_depth": {"min": 3, "max": 12, "type": "int"},
        "lgb_min_child_samples": {"min": 10, "max": 50, "type": "int"},
        "lgb_num_leaves": {"min": 20, "max": 100, "type": "int"},
        
        # Neural Networks
        "nn_learning_rate": {"min": 0.0001, "max": 0.01, "type": "float"},
        "nn_max_iter": {"min": 200, "max": 1000, "type": "int"},
        
        # Random Forest
        "rf_max_depth": {"min": 5, "max": 20, "type": "int"},
        "rf_min_samples_split": {"min": 2, "max": 10, "type": "int"},
        "rf_min_samples_leaf": {"min": 1, "max": 10, "type": "int"},
        "rf_n_estimators": {"min": 50, "max": 200, "type": "int"},
        
        # Step 11: Confidence Calibration
        "calibration_accuracy": {"min": 0.6, "max": 0.85, "type": "float"},
        "calibration_time_minutes": {"min": 30.0, "max": 120.0, "type": "float"},
        
        # Performance thresholds
        "model_performance_threshold": {"min": 0.6, "max": 0.85, "type": "float"},
        "data_quality_threshold": {"min": 0.9, "max": 0.99, "type": "float"},
        "artifact_completeness_threshold": {"min": 0.8, "max": 0.95, "type": "float"},
        
        # Memory and performance
        "memory_threshold_gb": {"min": 6.0, "max": 16.0, "type": "float"},
        "cpu_threshold_percent": {"min": 70.0, "max": 95.0, "type": "float"},
        "disk_threshold_gb": {"min": 3.0, "max": 10.0, "type": "float"},
        "monitor_interval": {"min": 15.0, "max": 60.0, "type": "float"},
        "failure_threshold": {"min": 2, "max": 5, "type": "int"},
    }