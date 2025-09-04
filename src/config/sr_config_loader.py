#!/usr/bin/env python3
"""S/R Configuration Loader.

This module loads and validates S/R optimization configuration from YAML files,
replacing hardcoded parameters throughout the codebase.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


@dataclass
class SRParameterRanges:
    """S/R parameter ranges for optimization."""
    # Method weights
    fractal_weight: Tuple[float, float] = (0.1, 0.6)
    volume_weight: Tuple[float, float] = (0.1, 0.5)
    pivot_weight: Tuple[float, float] = (0.1, 0.4)
    atr_weight: Tuple[float, float] = (0.05, 0.3)
    
    # Strength calculation weights
    touch_count_weight: Tuple[float, float] = (0.2, 0.5)
    total_volume_weight: Tuple[float, float] = (0.1, 0.4)
    level_age_weight: Tuple[float, float] = (0.1, 0.4)
    bounce_rate_weight: Tuple[float, float] = (0.1, 0.4)
    isolation_score_weight: Tuple[float, float] = (0.05, 0.3)
    
    # DBSCAN parameters
    dbscan_eps: Tuple[float, float] = (0.002, 0.025)
    dbscan_min_samples: Tuple[int, int] = (2, 6)
    
    # Multi-timeframe weights
    tf_15m_weight: Tuple[float, float] = (0.15, 0.3)
    tf_1h_weight: Tuple[float, float] = (0.2, 0.35)
    tf_4h_weight: Tuple[float, float] = (0.15, 0.3)
    tf_1d_weight: Tuple[float, float] = (0.05, 0.2)
    
    # Advanced method parameters
    fibonacci_sensitivity: Tuple[float, float] = (0.5, 0.9)
    elliott_confidence_threshold: Tuple[float, float] = (0.4, 0.8)
    order_flow_hvn_threshold: Tuple[float, float] = (1.1, 2.0)


@dataclass
class SRTimeframeConfig:
    """S/R configuration for specific timeframe."""
    touch_threshold: float
    bounce_threshold: float
    breakout_threshold: float
    min_touches: int
    volume_spike_threshold: float
    fractal_period: int
    pivot_period: int


@dataclass
class SRPerformanceThresholds:
    """S/R performance thresholds."""
    min_sr_validation_score: float = 0.6
    min_bounce_rate: float = 0.5
    max_false_breakout_rate: float = 0.4
    min_volume_confirmation: float = 0.4
    min_level_detection_accuracy: float = 0.3


@dataclass
class SROptimizationConfig:
    """Complete S/R optimization configuration."""
    n_trials: int
    cv_folds: int
    test_size: float
    optimization_timeout: int
    performance_thresholds: SRPerformanceThresholds
    timeframe_configs: Dict[str, SRTimeframeConfig]
    parameter_ranges: SRParameterRanges
    caching_config: Dict[str, Any]
    error_handling_config: Dict[str, Any]
    memory_management_config: Dict[str, Any]


class SRConfigLoader:
    """Loads and validates S/R optimization configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader."""
        self.logger = system_logger.getChild("SRConfigLoader")
        self.config_path = config_path or "src/config/sr_optimization_config.yaml"
        self._config: Optional[SROptimizationConfig] = None
    
    @handles_errors(
        exceptions=(FileNotFoundError, yaml.YAMLError, ValueError),
        default_return=None,
        context="load S/R configuration"
    )
    def load_config(self) -> Optional[SROptimizationConfig]:
        """Load S/R optimization configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.error(f"Configuration file not found: {self.config_path}")
                return None
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                self.logger.error("Empty configuration file")
                return None
            
            # Parse configuration
            self._config = self._parse_config(config_data)
            
            if self._config:
                self.logger.info("✅ S/R configuration loaded successfully")
                self._validate_config(self._config)
            else:
                self.logger.error("Failed to parse configuration")
            
            return self._config
            
        except Exception as e:
            self.logger.error(f"Failed to load S/R configuration: {e}")
            return None
    
    def _parse_config(self, config_data: Dict[str, Any]) -> Optional[SROptimizationConfig]:
        """Parse configuration data into structured objects."""
        try:
            # Parse optimization config
            opt_config = config_data.get("sr_detection_optimization", {})
            
            # Parse performance thresholds
            perf_thresholds_data = opt_config.get("performance_thresholds", {})
            performance_thresholds = SRPerformanceThresholds(
                min_sr_validation_score=perf_thresholds_data.get("min_sr_validation_score", 0.6),
                min_bounce_rate=perf_thresholds_data.get("min_bounce_rate", 0.5),
                max_false_breakout_rate=perf_thresholds_data.get("max_false_breakout_rate", 0.4),
                min_volume_confirmation=perf_thresholds_data.get("min_volume_confirmation", 0.4),
                min_level_detection_accuracy=perf_thresholds_data.get("min_level_detection_accuracy", 0.3)
            )
            
            # Parse timeframe configurations
            timeframe_configs = {}
            timeframe_data = opt_config.get("timeframe_config", {})
            for tf, tf_config in timeframe_data.items():
                timeframe_configs[tf] = SRTimeframeConfig(
                    touch_threshold=tf_config.get("touch_threshold", 0.001),
                    bounce_threshold=tf_config.get("bounce_threshold", 0.003),
                    breakout_threshold=tf_config.get("breakout_threshold", 0.008),
                    min_touches=tf_config.get("min_touches", 3),
                    volume_spike_threshold=tf_config.get("volume_spike_threshold", 1.5),
                    fractal_period=tf_config.get("fractal_period", 5),
                    pivot_period=tf_config.get("pivot_period", 10)
                )
            
            # Parse parameter ranges
            param_ranges_data = opt_config.get("parameter_ranges", {})
            parameter_ranges = SRParameterRanges(
                fractal_weight=tuple(param_ranges_data.get("fractal_weight", [0.1, 0.6])),
                volume_weight=tuple(param_ranges_data.get("volume_weight", [0.1, 0.5])),
                pivot_weight=tuple(param_ranges_data.get("pivot_weight", [0.1, 0.4])),
                atr_weight=tuple(param_ranges_data.get("atr_weight", [0.05, 0.3])),
                touch_count_weight=tuple(param_ranges_data.get("touch_count_weight", [0.2, 0.5])),
                total_volume_weight=tuple(param_ranges_data.get("total_volume_weight", [0.1, 0.4])),
                level_age_weight=tuple(param_ranges_data.get("level_age_weight", [0.1, 0.4])),
                bounce_rate_weight=tuple(param_ranges_data.get("bounce_rate_weight", [0.1, 0.4])),
                isolation_score_weight=tuple(param_ranges_data.get("isolation_score_weight", [0.05, 0.3])),
                tf_15m_weight=tuple(param_ranges_data.get("timeframe_weights", {}).get("tf_15m_weight", [0.15, 0.3])),
                tf_1h_weight=tuple(param_ranges_data.get("timeframe_weights", {}).get("tf_1h_weight", [0.2, 0.35])),
                tf_4h_weight=tuple(param_ranges_data.get("timeframe_weights", {}).get("tf_4h_weight", [0.15, 0.3])),
                tf_1d_weight=tuple(param_ranges_data.get("timeframe_weights", {}).get("tf_1d_weight", [0.05, 0.2])),
                fibonacci_sensitivity=tuple(param_ranges_data.get("fibonacci_sensitivity", [0.5, 0.9])),
                elliott_confidence_threshold=tuple(param_ranges_data.get("elliott_confidence_threshold", [0.4, 0.8])),
                order_flow_hvn_threshold=tuple(param_ranges_data.get("order_flow_hvn_threshold", [1.1, 2.0]))
            )
            
            # Parse additional configurations
            caching_config = config_data.get("caching", {})
            error_handling_config = config_data.get("error_handling", {})
            memory_management_config = config_data.get("memory_management", {})
            
            return SROptimizationConfig(
                n_trials=opt_config.get("n_trials", 100),
                cv_folds=opt_config.get("cv_folds", 5),
                test_size=opt_config.get("test_size", 0.2),
                optimization_timeout=opt_config.get("optimization_timeout", 3600),
                performance_thresholds=performance_thresholds,
                timeframe_configs=timeframe_configs,
                parameter_ranges=parameter_ranges,
                caching_config=caching_config,
                error_handling_config=error_handling_config,
                memory_management_config=memory_management_config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {e}")
            return None
    
    def _validate_config(self, config: SROptimizationConfig) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate optimization parameters
            if config.n_trials <= 0:
                raise ValueError("n_trials must be positive")
            if config.cv_folds < 2:
                raise ValueError("cv_folds must be at least 2")
            if not 0 < config.test_size < 1:
                raise ValueError("test_size must be between 0 and 1")
            if config.optimization_timeout <= 0:
                raise ValueError("optimization_timeout must be positive")
            
            # Validate performance thresholds
            thresholds = config.performance_thresholds
            if not 0 <= thresholds.min_sr_validation_score <= 1:
                raise ValueError("min_sr_validation_score must be between 0 and 1")
            if not 0 <= thresholds.min_bounce_rate <= 1:
                raise ValueError("min_bounce_rate must be between 0 and 1")
            if not 0 <= thresholds.max_false_breakout_rate <= 1:
                raise ValueError("max_false_breakout_rate must be between 0 and 1")
            
            # Validate timeframe configurations
            for tf, tf_config in config.timeframe_configs.items():
                if tf_config.touch_threshold <= 0:
                    raise ValueError(f"touch_threshold for {tf} must be positive")
                if tf_config.min_touches < 1:
                    raise ValueError(f"min_touches for {tf} must be at least 1")
                if tf_config.volume_spike_threshold < 1:
                    raise ValueError(f"volume_spike_threshold for {tf} must be at least 1")
            
            # Validate parameter ranges
            ranges = config.parameter_ranges
            for attr_name in dir(ranges):
                if not attr_name.startswith('_'):
                    attr_value = getattr(ranges, attr_name)
                    if isinstance(attr_value, tuple) and len(attr_value) == 2:
                        min_val, max_val = attr_value
                        if min_val >= max_val:
                            raise ValueError(f"Parameter range {attr_name}: min ({min_val}) must be less than max ({max_val})")
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config(self) -> Optional[SROptimizationConfig]:
        """Get loaded configuration."""
        return self._config
    
    def get_timeframe_config(self, timeframe: str) -> Optional[SRTimeframeConfig]:
        """Get configuration for specific timeframe."""
        if not self._config:
            return None
        return self._config.timeframe_configs.get(timeframe)
    
    def get_parameter_ranges(self) -> Optional[SRParameterRanges]:
        """Get parameter ranges for optimization."""
        if not self._config:
            return None
        return self._config.parameter_ranges
    
    def get_performance_thresholds(self) -> Optional[SRPerformanceThresholds]:
        """Get performance thresholds."""
        if not self._config:
            return None
        return self._config.performance_thresholds
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            if not self._config:
                self.logger.warning("No configuration loaded to update")
                return False
            
            # Update optimization parameters
            if "n_trials" in updates:
                self._config.n_trials = updates["n_trials"]
            if "cv_folds" in updates:
                self._config.cv_folds = updates["cv_folds"]
            if "test_size" in updates:
                self._config.test_size = updates["test_size"]
            
            # Update performance thresholds
            if "performance_thresholds" in updates:
                perf_updates = updates["performance_thresholds"]
                thresholds = self._config.performance_thresholds
                for key, value in perf_updates.items():
                    if hasattr(thresholds, key):
                        setattr(thresholds, key, value)
            
            # Validate updated configuration
            if self._validate_config(self._config):
                self.logger.info("✅ Configuration updated successfully")
                return True
            else:
                self.logger.error("Configuration update validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, output_path: Optional[str] = None) -> bool:
        """Save current configuration to YAML file."""
        try:
            if not self._config:
                self.logger.warning("No configuration to save")
                return False
            
            output_file = Path(output_path or self.config_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert configuration back to dictionary
            config_dict = self._config_to_dict(self._config)
            
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"✅ Configuration saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self, config: SROptimizationConfig) -> Dict[str, Any]:
        """Convert configuration object back to dictionary."""
        return {
            "sr_detection_optimization": {
                "n_trials": config.n_trials,
                "cv_folds": config.cv_folds,
                "test_size": config.test_size,
                "optimization_timeout": config.optimization_timeout,
                "performance_thresholds": {
                    "min_sr_validation_score": config.performance_thresholds.min_sr_validation_score,
                    "min_bounce_rate": config.performance_thresholds.min_bounce_rate,
                    "max_false_breakout_rate": config.performance_thresholds.max_false_breakout_rate,
                    "min_volume_confirmation": config.performance_thresholds.min_volume_confirmation,
                    "min_level_detection_accuracy": config.performance_thresholds.min_level_detection_accuracy
                },
                "timeframe_config": {
                    tf: {
                        "touch_threshold": tf_config.touch_threshold,
                        "bounce_threshold": tf_config.bounce_threshold,
                        "breakout_threshold": tf_config.breakout_threshold,
                        "min_touches": tf_config.min_touches,
                        "volume_spike_threshold": tf_config.volume_spike_threshold,
                        "fractal_period": tf_config.fractal_period,
                        "pivot_period": tf_config.pivot_period
                    }
                    for tf, tf_config in config.timeframe_configs.items()
                },
                "parameter_ranges": {
                    "fractal_weight": list(config.parameter_ranges.fractal_weight),
                    "volume_weight": list(config.parameter_ranges.volume_weight),
                    "pivot_weight": list(config.parameter_ranges.pivot_weight),
                    "atr_weight": list(config.parameter_ranges.atr_weight),
                    "touch_count_weight": list(config.parameter_ranges.touch_count_weight),
                    "total_volume_weight": list(config.parameter_ranges.total_volume_weight),
                    "level_age_weight": list(config.parameter_ranges.level_age_weight),
                    "bounce_rate_weight": list(config.parameter_ranges.bounce_rate_weight),
                    "isolation_score_weight": list(config.parameter_ranges.isolation_score_weight),
                    "timeframe_weights": {
                        "tf_15m_weight": list(config.parameter_ranges.tf_15m_weight),
                        "tf_1h_weight": list(config.parameter_ranges.tf_1h_weight),
                        "tf_4h_weight": list(config.parameter_ranges.tf_4h_weight),
                        "tf_1d_weight": list(config.parameter_ranges.tf_1d_weight)
                    },
                    "fibonacci_sensitivity": list(config.parameter_ranges.fibonacci_sensitivity),
                    "elliott_confidence_threshold": list(config.parameter_ranges.elliott_confidence_threshold),
                    "order_flow_hvn_threshold": list(config.parameter_ranges.order_flow_hvn_threshold)
                }
            },
            "caching": config.caching_config,
            "error_handling": config.error_handling_config,
            "memory_management": config.memory_management_config
        }


# Global configuration instance
_config_loader: Optional[SRConfigLoader] = None


def get_sr_config_loader() -> SRConfigLoader:
    """Get global S/R configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = SRConfigLoader()
        _config_loader.load_config()
    return _config_loader


def get_sr_config() -> Optional[SROptimizationConfig]:
    """Get S/R optimization configuration."""
    loader = get_sr_config_loader()
    return loader.get_config()