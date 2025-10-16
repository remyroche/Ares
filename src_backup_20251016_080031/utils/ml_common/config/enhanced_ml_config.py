#!/usr/bin/env python3
"""
Enhanced ML Pipeline Configuration

This module provides comprehensive configuration management for the enhanced
ML training, HPO, and testing pipelines.
"""

from typing import Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from src.common.config.loader import save_to_file as _unified_save_to_file, load_from_file as _unified_load_from_file

from src.utils.logger import get_logger

logger = get_logger("EnhancedMLConfig")

@dataclass
class ErrorDetectionConfig:
    """Configuration for error detection system."""
    enable_real_time_monitoring: bool = True
    alert_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'critical_errors_per_hour': 5,
        'high_errors_per_hour': 20,
        'same_error_repetition': 10,
        'component_failure_rate': 0.3
    })
    classification_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retention_days: int = 30

@dataclass
class HPOMonitoringConfig:
    """Configuration for HPO monitoring."""
    convergence: Dict[str, Any] = field(default_factory=lambda: {
        'improvement_threshold': 0.001,
        'patience_trials': 20,
        'variance_threshold': 0.01,
        'confidence_level': 0.95,
        'min_trials_for_convergence': 10
    })
    failure_detection: Dict[str, Any] = field(default_factory=lambda: {
        'max_failure_rate': 0.3,
        'consecutive_failures_threshold': 5,
        'timeout_threshold': 3600,  # 1 hour
        'memory_threshold': 0.9,  # 90% of available memory
        'performance_degradation_threshold': 0.1
    })
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        'enable_early_stopping': True,
        'patience': 15,
        'min_delta': 0.001,
        'restore_best_weights': True,
        'monitor': 'objective_value'
    })

@dataclass
class TestingConfig:
    """Configuration for testing framework."""
    test_dir: str = "tests"
    output_dir: str = "test_results"
    parallel_workers: int = 4
    timeout_per_test: float = 300.0
    timeout_per_suite: float = 1800.0
    retry_failed_tests: bool = True
    max_retries: int = 3
    generate_reports: bool = True
    save_artifacts: bool = True
    coverage_analysis: bool = True
    performance_benchmarks: bool = True

@dataclass
class ReportingConfig:
    """Configuration for reporting system."""
    output_dir: str = "reports"
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enable_real_time': True,
        'report_interval': 300,  # 5 minutes
        'alert_check_interval': 60,  # 1 minute
        'retention_days': 30
    })
    notifications: Dict[str, Any] = field(default_factory=lambda: {
        'email_enabled': False,
        'slack_enabled': False,
        'webhook_enabled': False
    })

@dataclass
class PipelineConfig:
    """Configuration for pipeline integration."""
    max_errors_per_pipeline: int = 10
    max_critical_errors_per_pipeline: int = 3
    stage_timeout: float = 3600.0  # 1 hour per stage
    enable_stage_validation: bool = True
    enable_automatic_retry: bool = True
    max_retry_attempts: int = 3

@dataclass
class EnhancedMLConfig:
    """Complete configuration for enhanced ML pipeline."""
    error_detection: ErrorDetectionConfig = field(default_factory=ErrorDetectionConfig)
    hpo_monitoring: HPOMonitoringConfig = field(default_factory=HPOMonitoringConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Global settings
    enable_all_components: bool = True
    log_level: str = "INFO"
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'error_detection': {
                'enable_real_time_monitoring': self.error_detection.enable_real_time_monitoring,
                'alert_thresholds': self.error_detection.alert_thresholds,
                'retention_days': self.error_detection.retention_days
            },
            'hpo_monitoring': {
                'convergence': self.hpo_monitoring.convergence,
                'failure_detection': self.hpo_monitoring.failure_detection,
                'early_stopping': self.hpo_monitoring.early_stopping
            },
            'testing': {
                'test_dir': self.testing.test_dir,
                'output_dir': self.testing.output_dir,
                'parallel_workers': self.testing.parallel_workers,
                'timeout_per_test': self.testing.timeout_per_test,
                'timeout_per_suite': self.testing.timeout_per_suite,
                'retry_failed_tests': self.testing.retry_failed_tests,
                'max_retries': self.testing.max_retries,
                'generate_reports': self.testing.generate_reports,
                'save_artifacts': self.testing.save_artifacts,
                'coverage_analysis': self.testing.coverage_analysis,
                'performance_benchmarks': self.testing.performance_benchmarks
            },
            'reporting': {
                'output_dir': self.reporting.output_dir,
                'monitoring': self.reporting.monitoring,
                'notifications': self.reporting.notifications
            },
            'pipeline': {
                'max_errors_per_pipeline': self.pipeline.max_errors_per_pipeline,
                'max_critical_errors_per_pipeline': self.pipeline.max_critical_errors_per_pipeline,
                'stage_timeout': self.pipeline.stage_timeout,
                'enable_stage_validation': self.pipeline.enable_stage_validation,
                'enable_automatic_retry': self.pipeline.enable_automatic_retry,
                'max_retry_attempts': self.pipeline.max_retry_attempts
            },
            'global_settings': {
                'enable_all_components': self.enable_all_components,
                'log_level': self.log_level,
                'debug_mode': self.debug_mode
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedMLConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Error detection config
        if 'error_detection' in config_dict:
            ed_config = config_dict['error_detection']
            config.error_detection = ErrorDetectionConfig(
                enable_real_time_monitoring=ed_config.get('enable_real_time_monitoring', True),
                alert_thresholds=ed_config.get('alert_thresholds', config.error_detection.alert_thresholds),
                retention_days=ed_config.get('retention_days', 30)
            )
        
        # HPO monitoring config
        if 'hpo_monitoring' in config_dict:
            hpo_config = config_dict['hpo_monitoring']
            config.hpo_monitoring = HPOMonitoringConfig(
                convergence=hpo_config.get('convergence', config.hpo_monitoring.convergence),
                failure_detection=hpo_config.get('failure_detection', config.hpo_monitoring.failure_detection),
                early_stopping=hpo_config.get('early_stopping', config.hpo_monitoring.early_stopping)
            )
        
        # Testing config
        if 'testing' in config_dict:
            test_config = config_dict['testing']
            config.testing = TestingConfig(
                test_dir=test_config.get('test_dir', 'tests'),
                output_dir=test_config.get('output_dir', 'test_results'),
                parallel_workers=test_config.get('parallel_workers', 4),
                timeout_per_test=test_config.get('timeout_per_test', 300.0),
                timeout_per_suite=test_config.get('timeout_per_suite', 1800.0),
                retry_failed_tests=test_config.get('retry_failed_tests', True),
                max_retries=test_config.get('max_retries', 3),
                generate_reports=test_config.get('generate_reports', True),
                save_artifacts=test_config.get('save_artifacts', True),
                coverage_analysis=test_config.get('coverage_analysis', True),
                performance_benchmarks=test_config.get('performance_benchmarks', True)
            )
        
        # Reporting config
        if 'reporting' in config_dict:
            report_config = config_dict['reporting']
            config.reporting = ReportingConfig(
                output_dir=report_config.get('output_dir', 'reports'),
                monitoring=report_config.get('monitoring', config.reporting.monitoring),
                notifications=report_config.get('notifications', config.reporting.notifications)
            )
        
        # Pipeline config
        if 'pipeline' in config_dict:
            pipe_config = config_dict['pipeline']
            config.pipeline = PipelineConfig(
                max_errors_per_pipeline=pipe_config.get('max_errors_per_pipeline', 10),
                max_critical_errors_per_pipeline=pipe_config.get('max_critical_errors_per_pipeline', 3),
                stage_timeout=pipe_config.get('stage_timeout', 3600.0),
                enable_stage_validation=pipe_config.get('enable_stage_validation', True),
                enable_automatic_retry=pipe_config.get('enable_automatic_retry', True),
                max_retry_attempts=pipe_config.get('max_retry_attempts', 3)
            )
        
        # Global settings
        if 'global_settings' in config_dict:
            global_config = config_dict['global_settings']
            config.enable_all_components = global_config.get('enable_all_components', True)
            config.log_level = global_config.get('log_level', 'INFO')
            config.debug_mode = global_config.get('debug_mode', False)
        
        return config
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to file using the unified loader."""
        try:
            _unified_save_to_file(self, filepath)
            logger.info(f"✅ Configuration saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'EnhancedMLConfig':
        """Load configuration from file using the unified loader."""
        try:
            config = _unified_load_from_file(filepath, cls)  # type: ignore[return-value]
            logger.info(f"✅ Configuration loaded from: {filepath}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

# Default configuration
DEFAULT_CONFIG = EnhancedMLConfig()

# Configuration presets
DEVELOPMENT_CONFIG = EnhancedMLConfig(
    error_detection=ErrorDetectionConfig(
        enable_real_time_monitoring=True,
        alert_thresholds={
            'critical_errors_per_hour': 3,
            'high_errors_per_hour': 10,
            'same_error_repetition': 5,
            'component_failure_rate': 0.2
        }
    ),
    hpo_monitoring=HPOMonitoringConfig(
        convergence={
            'improvement_threshold': 0.01,
            'patience_trials': 10,
            'variance_threshold': 0.05,
            'confidence_level': 0.9,
            'min_trials_for_convergence': 5
        }
    ),
    testing=TestingConfig(
        timeout_per_test=60.0,
        timeout_per_suite=300.0,
        max_retries=1
    ),
    pipeline=PipelineConfig(
        max_errors_per_pipeline=5,
        max_critical_errors_per_pipeline=1
    ),
    debug_mode=True
)

PRODUCTION_CONFIG = EnhancedMLConfig(
    error_detection=ErrorDetectionConfig(
        enable_real_time_monitoring=True,
        alert_thresholds={
            'critical_errors_per_hour': 1,
            'high_errors_per_hour': 5,
            'same_error_repetition': 3,
            'component_failure_rate': 0.1
        }
    ),
    hpo_monitoring=HPOMonitoringConfig(
        convergence={
            'improvement_threshold': 0.001,
            'patience_trials': 30,
            'variance_threshold': 0.01,
            'confidence_level': 0.95,
            'min_trials_for_convergence': 15
        }
    ),
    testing=TestingConfig(
        timeout_per_test=600.0,
        timeout_per_suite=3600.0,
        max_retries=3
    ),
    pipeline=PipelineConfig(
        max_errors_per_pipeline=3,
        max_critical_errors_per_pipeline=1
    ),
    debug_mode=False
)

def get_config(preset: str = "default") -> EnhancedMLConfig:
    """Get configuration preset."""
    presets = {
        "default": DEFAULT_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]