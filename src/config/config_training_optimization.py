from __future__ import annotations
# src/config/config_training_optimization.py

"""
Configuration file for optimizable training optimization parameters from other steps.
These parameters can be optimized in step12.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingOptimizationConfig:
    """Optimizable training optimization parameters from other steps."""

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
    architecture_optimization_enabled: bool = False

    # Step 6: Analyst Enhancement
    stability_threshold: float = 0.7
    mi_threshold: float = 0.01
    feature_selection_threshold: float = 0.2

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
        "architecture_optimization_enabled": {"type": "bool"},

        # Step 6: Analyst Enhancement
        "stability_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "mi_threshold": {"min": 0.005, "max": 0.02, "type": "float"},
        "feature_selection_threshold": {"min": 0.1, "max": 0.4, "type": "float"},

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
