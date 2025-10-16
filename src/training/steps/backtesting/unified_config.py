"""
Unified Configuration System for Backtesting

This module provides a unified configuration system that eliminates duplication
across all backtesting components and provides a builder pattern for easy configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from copy import deepcopy

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution modes for backtesting components."""
    BLANK = "blank"        # Minimal execution for testing/validation
    LIGHT = "light"        # Lightweight execution with essential features only
    FULL = "full"          # Complete execution with all features

class OptimizationMethod(Enum):
    """Optimization methods for parameter optimization."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    GRADIENT_DESCENT = "gradient_descent"

class MonteCarloMode(Enum):
    """Monte Carlo simulation modes."""
    BOOTSTRAP = "bootstrap"
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    HYBRID = "hybrid"

class ABTestType(Enum):
    """A/B test types."""
    PERFORMANCE = "performance"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    COMPREHENSIVE = "comprehensive"

class ReportType(Enum):
    """Report types."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"

@dataclass
class HardwareConfig:
    """Hardware optimization configuration."""
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    gpu_memory_limit: Optional[float] = None
    cpu_cores: Optional[int] = None

@dataclass
class DataConfig:
    """Data configuration."""
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    data_dir: str = "/workspace/data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_type: str = "processed"  # "raw", "processed", "optimized"
    cache_enabled: bool = True
    compression: str = "snappy"  # "snappy", "gzip", "lz4"

@dataclass
class ValidationConfig:
    """Validation and monitoring configuration."""
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    enable_cv_validation: bool = True
    enable_hpo: bool = True
    cv_folds: int = 5
    cv_method: str = "purged"  # "purged", "blocking", "standard"
    lookahead_bias_protection: bool = True
    overfitting_detection: bool = True

@dataclass
class BacktestingConfig:
    """Backtesting-specific configuration."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.1
    min_position_size: float = 0.01
    rebalance_frequency: str = 'daily'
    risk_free_rate: float = 0.02
    max_drawdown: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.1
    capacity_limit: float = 1.0
    market_impact_coefficient: float = 0.0005
    turnover_warning_threshold: float = 0.8

@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration."""
    n_simulations: int = 1000
    confidence_level: float = 0.95
    simulation_horizon: int = 252
    mode: MonteCarloMode = MonteCarloMode.HYBRID
    bootstrap_sample_size: float = 0.8
    parametric_distribution: str = "normal"
    var_confidence: float = 0.05
    expected_shortfall_confidence: float = 0.01

@dataclass
class ABTestConfig:
    """A/B testing configuration."""
    test_type: ABTestType = ABTestType.COMPREHENSIVE
    significance_level: float = 0.05
    power: float = 0.8
    min_sample_size: int = 30
    test_duration_days: int = 252
    warmup_period_days: int = 30
    cooldown_period_days: int = 7
    multiple_comparison_correction: str = "bonferroni"
    effect_size_threshold: float = 0.1
    confidence_interval: float = 0.95

@dataclass
class OptimizationConfig:
    """Parameter optimization configuration."""
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN
    n_trials: int = 100
    timeout_seconds: int = 3600
    early_stopping_patience: int = 10
    convergence_threshold: float = 1e-6
    objective_metric: str = "sharpe_ratio"
    minimize_objective: bool = False
    hpo_method: str = "bayesian"

@dataclass
class ReportingConfig:
    """Reporting configuration."""
    report_type: ReportType = ReportType.COMPREHENSIVE
    output_dir: str = "reports"
    output_format: str = "html"
    enable_plots: bool = True
    plot_style: str = "seaborn"
    figure_size: tuple = (12, 8)
    dpi: int = 300
    include_performance_metrics: bool = True
    include_risk_analysis: bool = True
    include_trade_analysis: bool = True
    include_portfolio_analysis: bool = True
    include_visualizations: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = False
    log_file: Optional[str] = None
    enable_debug: bool = False
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class TrailingTPConfig:
    """Configuration for trailing take-profit simulations."""

    activation_rr: float = 1.2
    trail_distance_pct: float = 0.01
    volatility_sensitivity: float = 1.0
    max_latency_seconds: int = 120
    noise_levels: List[float] = field(default_factory=lambda: [0.0005, 0.001])

@dataclass
class ScenarioSweepConfig:
    """Configuration for volatility scenario sweeps."""

    scenarios: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            'low': {
                'trail_multiplier': 0.8,
                'activation_multiplier': 1.1,
                'tp_multiplier': 0.9,
                'volatility_threshold': 0.01,
                'latency_buffer_seconds': 5,
                'noise_levels': [0.0003, 0.0006],
            },
            'normal': {
                'trail_multiplier': 1.0,
                'activation_multiplier': 1.0,
                'tp_multiplier': 1.0,
                'volatility_threshold': 0.02,
                'latency_buffer_seconds': 10,
                'noise_levels': [0.0005, 0.001],
            },
            'high': {
                'trail_multiplier': 1.3,
                'activation_multiplier': 0.9,
                'tp_multiplier': 1.15,
                'volatility_threshold': 0.035,
                'latency_buffer_seconds': 20,
                'noise_levels': [0.001, 0.0015],
            },
        }
    )

@dataclass
class UnifiedBacktestingConfig:
    """
    Unified configuration for all backtesting components.

    This configuration eliminates duplication by centralizing all common
    parameters and providing component-specific configurations.
    """
    # Core configuration
    mode: ExecutionMode = ExecutionMode.FULL
    force_rerun: bool = False
    single_stage_only: bool = False

    # Component configurations
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    ab_testing: ABTestConfig = field(default_factory=ABTestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trailing_tp: TrailingTPConfig = field(default_factory=TrailingTPConfig)
    scenario_sweep: ScenarioSweepConfig = field(default_factory=ScenarioSweepConfig)

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    description: Optional[str] = None

class ConfigurationBuilder:
    """
    Builder pattern for creating unified backtesting configurations.

    This builder provides a fluent interface for creating configurations
    with sensible defaults and easy customization.
    """

    def __init__(self):
        """Initialize the configuration builder."""
        self._config = UnifiedBacktestingConfig()
        self._component_configs = {}
        self._ensure_custom_defaults()

    def _ensure_custom_defaults(self) -> None:
        """Ensure trailing TP and scenario defaults are present in custom params."""
        trailing_defaults = {
            'activation_rr': self._config.trailing_tp.activation_rr,
            'trail_distance_pct': self._config.trailing_tp.trail_distance_pct,
            'volatility_sensitivity': self._config.trailing_tp.volatility_sensitivity,
            'max_latency_seconds': self._config.trailing_tp.max_latency_seconds,
            'noise_levels': list(self._config.trailing_tp.noise_levels),
        }

        existing_trailing = self._config.custom_params.get('trailing_tp', {})
        merged_trailing = {**trailing_defaults, **existing_trailing}
        self._config.custom_params['trailing_tp'] = merged_trailing

        scenario_defaults = {
            name: deepcopy(cfg)
            for name, cfg in self._config.scenario_sweep.scenarios.items()
        }

        existing_scenarios = self._config.custom_params.get('volatility_scenarios', {})
        merged_scenarios: Dict[str, Dict[str, Any]] = {}

        for name, default_cfg in scenario_defaults.items():
            custom_cfg = existing_scenarios.get(name, {}) if isinstance(existing_scenarios, dict) else {}
            merged_scenarios[name] = {**default_cfg, **custom_cfg}

        if isinstance(existing_scenarios, dict):
            for name, custom_cfg in existing_scenarios.items():
                if name not in merged_scenarios:
                    merged_scenarios[name] = custom_cfg

        self._config.custom_params['volatility_scenarios'] = merged_scenarios

    def set_mode(self, mode: ExecutionMode) -> 'ConfigurationBuilder':
        """Set the execution mode."""
        self._config.mode = mode
        return self

    def set_symbol(self, symbol: str) -> 'ConfigurationBuilder':
        """Set the trading symbol."""
        self._config.data.symbol = symbol
        return self

    def set_exchange(self, exchange: str) -> 'ConfigurationBuilder':
        """Set the exchange."""
        self._config.data.exchange = exchange
        return self

    def set_timeframe(self, timeframe: str) -> 'ConfigurationBuilder':
        """Set the timeframe."""
        self._config.data.timeframe = timeframe
        return self

    def set_data_dir(self, data_dir: str) -> 'ConfigurationBuilder':
        """Set the data directory."""
        self._config.data.data_dir = data_dir
        return self

    def set_date_range(self, start_date: str, end_date: str) -> 'ConfigurationBuilder':
        """Set the date range."""
        self._config.data.start_date = start_date
        self._config.data.end_date = end_date
        return self

    def set_hardware_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set hardware configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.hardware, key):
                setattr(self._config.hardware, key, value)
        return self

    def set_validation_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set validation configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.validation, key):
                setattr(self._config.validation, key, value)
        return self

    def set_backtesting_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set backtesting configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.backtesting, key):
                setattr(self._config.backtesting, key, value)
        return self

    def set_monte_carlo_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set Monte Carlo configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.monte_carlo, key):
                setattr(self._config.monte_carlo, key, value)
        return self

    def set_ab_testing_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set A/B testing configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.ab_testing, key):
                setattr(self._config.ab_testing, key, value)
        return self

    def set_optimization_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set optimization configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.optimization, key):
                setattr(self._config.optimization, key, value)
        return self

    def set_reporting_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set reporting configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.reporting, key):
                setattr(self._config.reporting, key, value)
        return self

    def set_logging_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Set logging configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.logging, key):
                setattr(self._config.logging, key, value)
        return self

    def set_custom_params(self, **kwargs) -> 'ConfigurationBuilder':
        """Set custom parameters."""
        self._config.custom_params.update(kwargs)
        self._ensure_custom_defaults()
        return self

    def set_trailing_tp_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Update trailing take-profit simulation configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config.trailing_tp, key):
                setattr(self._config.trailing_tp, key, value)

        self._ensure_custom_defaults()
        self._config.custom_params['trailing_tp'].update(kwargs)
        return self

    def set_scenario_sweep_config(self, scenarios: Dict[str, Dict[str, Any]]) -> 'ConfigurationBuilder':
        """Set volatility scenario sweep configuration."""
        if not isinstance(scenarios, dict):
            raise ValueError("scenarios must be a dictionary")

        self._config.scenario_sweep.scenarios = scenarios
        self._ensure_custom_defaults()
        return self

    def enable_gpu_acceleration(self, enabled: bool = True) -> 'ConfigurationBuilder':
        """Enable/disable GPU acceleration."""
        self._config.hardware.enable_gpu_acceleration = enabled
        return self

    def enable_parallel_processing(self, enabled: bool = True, max_workers: int = 4) -> 'ConfigurationBuilder':
        """Enable/disable parallel processing."""
        self._config.hardware.enable_parallel_processing = enabled
        self._config.hardware.max_workers = max_workers
        return self

    def enable_validation(self, enabled: bool = True) -> 'ConfigurationBuilder':
        """Enable/disable validation."""
        self._config.validation.validation_enabled = enabled
        return self

    def enable_monitoring(self, enabled: bool = True) -> 'ConfigurationBuilder':
        """Enable/disable monitoring."""
        self._config.validation.monitoring_enabled = enabled
        return self

    def set_initial_capital(self, capital: float) -> 'ConfigurationBuilder':
        """Set initial capital."""
        self._config.backtesting.initial_capital = capital
        return self

    def set_commission_rate(self, rate: float) -> 'ConfigurationBuilder':
        """Set commission rate."""
        self._config.backtesting.commission_rate = rate
        return self

    def set_slippage_rate(self, rate: float) -> 'ConfigurationBuilder':
        """Set slippage rate."""
        self._config.backtesting.slippage_rate = rate
        return self

    def set_n_simulations(self, n: int) -> 'ConfigurationBuilder':
        """Set number of Monte Carlo simulations."""
        self._config.monte_carlo.n_simulations = n
        return self

    def set_confidence_level(self, level: float) -> 'ConfigurationBuilder':
        """Set confidence level."""
        self._config.monte_carlo.confidence_level = level
        return self

    def set_significance_level(self, level: float) -> 'ConfigurationBuilder':
        """Set significance level for A/B testing."""
        self._config.ab_testing.significance_level = level
        return self

    def set_n_trials(self, n: int) -> 'ConfigurationBuilder':
        """Set number of optimization trials."""
        self._config.optimization.n_trials = n
        return self

    def set_output_dir(self, directory: str) -> 'ConfigurationBuilder':
        """Set output directory."""
        self._config.reporting.output_dir = directory
        return self

    def set_output_format(self, format: str) -> 'ConfigurationBuilder':
        """Set output format."""
        self._config.reporting.output_format = format
        return self

    def for_testing(self) -> 'ConfigurationBuilder':
        """Configure for testing (BLANK mode with minimal resources)."""
        self._config.mode = ExecutionMode.BLANK
        self._config.hardware.enable_gpu_acceleration = False
        self._config.hardware.enable_parallel_processing = False
        self._config.hardware.max_workers = 1
        self._config.backtesting.initial_capital = 10000.0
        self._config.monte_carlo.n_simulations = 100
        self._config.optimization.n_trials = 10
        return self

    def for_development(self) -> 'ConfigurationBuilder':
        """Configure for development (LIGHT mode with moderate resources)."""
        self._config.mode = ExecutionMode.LIGHT
        self._config.hardware.enable_gpu_acceleration = True
        self._config.hardware.enable_parallel_processing = True
        self._config.hardware.max_workers = 2
        self._config.backtesting.initial_capital = 50000.0
        self._config.monte_carlo.n_simulations = 500
        self._config.optimization.n_trials = 50
        return self

    def for_production(self) -> 'ConfigurationBuilder':
        """Configure for production (FULL mode with all resources)."""
        self._config.mode = ExecutionMode.FULL
        self._config.hardware.enable_gpu_acceleration = True
        self._config.hardware.enable_memory_optimization = True
        self._config.hardware.enable_parallel_processing = True
        self._config.hardware.max_workers = 4
        self._config.backtesting.initial_capital = 100000.0
        self._config.monte_carlo.n_simulations = 1000
        self._config.optimization.n_trials = 100
        return self

    def build(self) -> UnifiedBacktestingConfig:
        """Build the final configuration."""
        # Validate configuration
        self._validate_config()

        # Set description if not provided
        if not self._config.description:
            self._config.description = f"Backtesting configuration for {self._config.data.symbol} on {self._config.data.exchange}"

        return self._config

    def _validate_config(self) -> None:
        """Validate the configuration."""
        try:
            # Validate data configuration
            if not self._config.data.symbol:
                raise ValueError("Symbol is required")
            if not self._config.data.exchange:
                raise ValueError("Exchange is required")
            if not self._config.data.timeframe:
                raise ValueError("Timeframe is required")

            # Validate hardware configuration
            if self._config.hardware.max_workers < 1:
                raise ValueError("max_workers must be at least 1")

            # Validate backtesting configuration
            if self._config.backtesting.initial_capital <= 0:
                raise ValueError("initial_capital must be positive")
            if not 0 <= self._config.backtesting.commission_rate <= 1:
                raise ValueError("commission_rate must be between 0 and 1")
            if not 0 <= self._config.backtesting.slippage_rate <= 1:
                raise ValueError("slippage_rate must be between 0 and 1")

            # Validate Monte Carlo configuration
            if self._config.monte_carlo.n_simulations < 1:
                raise ValueError("n_simulations must be at least 1")
            if not 0 < self._config.monte_carlo.confidence_level < 1:
                raise ValueError("confidence_level must be between 0 and 1")

            # Validate A/B testing configuration
            if not 0 < self._config.ab_testing.significance_level < 1:
                raise ValueError("significance_level must be between 0 and 1")
            if not 0 < self._config.ab_testing.power < 1:
                raise ValueError("power must be between 0 and 1")

            # Validate optimization configuration
            if self._config.optimization.n_trials < 1:
                raise ValueError("n_trials must be at least 1")

            logger.info("✅ Configuration validation passed")

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise

# Convenience functions
def create_config() -> ConfigurationBuilder:
    """Create a new configuration builder."""
    return ConfigurationBuilder()

def create_testing_config() -> UnifiedBacktestingConfig:
    """Create a configuration optimized for testing."""
    return ConfigurationBuilder().for_testing().build()

def create_development_config() -> UnifiedBacktestingConfig:
    """Create a configuration optimized for development."""
    return ConfigurationBuilder().for_development().build()

def create_production_config() -> UnifiedBacktestingConfig:
    """Create a configuration optimized for production."""
    return ConfigurationBuilder().for_production().build()

def create_custom_config(**kwargs) -> UnifiedBacktestingConfig:
    """Create a custom configuration."""
    builder = ConfigurationBuilder()

    # Apply custom parameters
    for key, value in kwargs.items():
        if hasattr(builder, f"set_{key}"):
            getattr(builder, f"set_{key}")(value)
        else:
            builder.set_custom_params(**{key: value})

    return builder.build()

# Configuration presets
class ConfigurationPresets:
    """Predefined configuration presets for common use cases."""

    @staticmethod
    def crypto_day_trading() -> UnifiedBacktestingConfig:
        """Configuration for crypto day trading."""
        return (ConfigurationBuilder()
                .set_symbol("BTCUSDT")
                .set_exchange("binance")
                .set_timeframe("1h")
                .set_initial_capital(100000.0)
                .set_commission_rate(0.001)
                .set_slippage_rate(0.0005)
                .for_production()
                .build())

    @staticmethod
    def crypto_swing_trading() -> UnifiedBacktestingConfig:
        """Configuration for crypto swing trading."""
        return (ConfigurationBuilder()
                .set_symbol("ETHUSDT")
                .set_exchange("binance")
                .set_timeframe("4h")
                .set_initial_capital(50000.0)
                .set_commission_rate(0.001)
                .set_slippage_rate(0.0005)
                .for_production()
                .build())

    @staticmethod
    def forex_scalping() -> UnifiedBacktestingConfig:
        """Configuration for forex scalping."""
        return (ConfigurationBuilder()
                .set_symbol("EURUSD")
                .set_exchange("oanda")
                .set_timeframe("1m")
                .set_initial_capital(10000.0)
                .set_commission_rate(0.0001)
                .set_slippage_rate(0.0001)
                .for_production()
                .build())

    @staticmethod
    def stock_swing_trading() -> UnifiedBacktestingConfig:
        """Configuration for stock swing trading."""
        return (ConfigurationBuilder()
                .set_symbol("AAPL")
                .set_exchange("yahoo")
                .set_timeframe("1d")
                .set_initial_capital(100000.0)
                .set_commission_rate(0.005)
                .set_slippage_rate(0.001)
                .for_production()
                .build())
