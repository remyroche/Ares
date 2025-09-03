from __future__ import annotations
'\nConfiguration file for optimizable system monitoring and performance parameters.\nThese parameters can be optimized in step12.\n'
from dataclasses import dataclass
from typing import Any

@dataclass
class SystemMonitoringConfig:
    """Optimizable system monitoring and performance parameters."""
    analysis_interval: int = 3600
    supervision_interval: int = 60
    optimization_interval: int = 300
    monitoring_interval: int = 30
    tracking_interval: int = 60
    report_interval: int = 300
    weighting_interval: int = 120
    allocation_interval: int = 180
    management_interval: int = 240
    check_interval: int = 45
    processing_interval: int = 15
    strategy_interval: int = 90
    max_analysis_history: int = 1000
    max_history: int = 100
    max_weighting_history: int = 500
    max_management_history: int = 200
    max_calculation_history: int = 300
    max_history_size: int = 1000
    max_strategy_history: int = 150
    enable_real_time_reporting: bool = True
    real_time_interval: int = 30
    enable_detailed_reporting: bool = True
    enable_performance_monitoring: bool = True
    enable_risk_metrics: bool = True
    enable_optimization_metrics: bool = True
    enable_model_monitoring: bool = True
    drift_check_interval: int = 300
    performance_snapshot_interval: int = 600
    feature_analysis_interval: int = 900
    prediction_drift_threshold: float = 0.05
    behavior_thresholds: dict[str, float] = None
    max_cache_size: int = 1000
    max_workers: int = 8
    memory_threshold: float = 0.8
    max_recovery_attempts: int = 3
    recovery_cooldown: int = 60
    failure_threshold: int = 5
    timeout: int = 60
    enable_batch_processing: bool = True
    enable_progress_tracking: bool = True
    enable_feature_caching: bool = True
    enable_wavelet_transforms: bool = True
    enable_persistence: bool = True
    enable_snapshots: bool = True
    snapshot_frequency: int = 3600
    export_formats: list[str] = None
    export_directory: str = 'exports'
    storage_path: str = 'storage'
    learning_rate: float = 0.01
    min_weight: float = 0.1
    max_weight: float = 0.8
    enable_adaptive_weighting: bool = True
    enable_performance_weighting: bool = True
    enable_risk_weighting: bool = True
    performance_multiplier_base: float = 0.5
    performance_multiplier_range: float = 1.0
    enable_portfolio_allocation: bool = True
    enable_risk_management: bool = True
    enable_rebalancing: bool = True

    def __post_init__(self) -> None:
        if self.behavior_thresholds is None:
            self.behavior_thresholds = {'consistency_threshold': 0.7, 'stability_threshold': 0.8, 'drift_threshold': 0.05}
        if self.export_formats is None:
            self.export_formats = ['csv', 'json', 'parquet']

def get_system_monitoring_config() -> SystemMonitoringConfig:
    """Get system monitoring configuration."""
    return SystemMonitoringConfig()

def get_system_monitoring_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for system monitoring optimization."""
    return {'analysis_interval': {'min': 1800, 'max': 7200, 'type': 'int'}, 'supervision_interval': {'min': 30, 'max': 120, 'type': 'int'}, 'optimization_interval': {'min': 180, 'max': 600, 'type': 'int'}, 'monitoring_interval': {'min': 15, 'max': 60, 'type': 'int'}, 'tracking_interval': {'min': 30, 'max': 120, 'type': 'int'}, 'report_interval': {'min': 180, 'max': 600, 'type': 'int'}, 'weighting_interval': {'min': 60, 'max': 300, 'type': 'int'}, 'allocation_interval': {'min': 120, 'max': 360, 'type': 'int'}, 'management_interval': {'min': 180, 'max': 480, 'type': 'int'}, 'check_interval': {'min': 30, 'max': 90, 'type': 'int'}, 'processing_interval': {'min': 10, 'max': 30, 'type': 'int'}, 'strategy_interval': {'min': 60, 'max': 180, 'type': 'int'}, 'max_analysis_history': {'min': 500, 'max': 2000, 'type': 'int'}, 'max_history': {'min': 50, 'max': 200, 'type': 'int'}, 'max_weighting_history': {'min': 250, 'max': 1000, 'type': 'int'}, 'max_management_history': {'min': 100, 'max': 400, 'type': 'int'}, 'max_calculation_history': {'min': 150, 'max': 600, 'type': 'int'}, 'max_history_size': {'min': 500, 'max': 2000, 'type': 'int'}, 'max_strategy_history': {'min': 75, 'max': 300, 'type': 'int'}, 'real_time_interval': {'min': 15, 'max': 60, 'type': 'int'}, 'drift_check_interval': {'min': 180, 'max': 600, 'type': 'int'}, 'performance_snapshot_interval': {'min': 300, 'max': 1200, 'type': 'int'}, 'feature_analysis_interval': {'min': 600, 'max': 1800, 'type': 'int'}, 'prediction_drift_threshold': {'min': 0.02, 'max': 0.1, 'type': 'float'}, 'max_cache_size': {'min': 500, 'max': 2000, 'type': 'int'}, 'max_workers': {'min': 4, 'max': 16, 'type': 'int'}, 'memory_threshold': {'min': 0.7, 'max': 0.9, 'type': 'float'}, 'max_recovery_attempts': {'min': 2, 'max': 5, 'type': 'int'}, 'recovery_cooldown': {'min': 30, 'max': 120, 'type': 'int'}, 'failure_threshold': {'min': 3, 'max': 8, 'type': 'int'}, 'timeout': {'min': 30, 'max': 120, 'type': 'int'}, 'snapshot_frequency': {'min': 1800, 'max': 7200, 'type': 'int'}, 'learning_rate': {'min': 0.005, 'max': 0.05, 'type': 'float'}, 'min_weight': {'min': 0.05, 'max': 0.2, 'type': 'float'}, 'max_weight': {'min': 0.7, 'max': 0.9, 'type': 'float'}, 'performance_multiplier_base': {'min': 0.3, 'max': 0.7, 'type': 'float'}, 'performance_multiplier_range': {'min': 0.5, 'max': 1.5, 'type': 'float'}}