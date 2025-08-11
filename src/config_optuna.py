# src/config_optuna.py

"""
Optuna Configuration for Strategy-Level Meta-Parameters

This file contains all the trading parameters that can be optimized during training.
These parameters are used throughout the codebase and should be referenced from this file
instead of being hardcoded in individual components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EnsembleMethod(Enum):
    """Enum for ensemble gathering methods."""

    ALL_THRESHOLD = "all_threshold"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    META_LEARNER = "meta_learner"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    REGIME_SPECIFIC = "regime_specific"


class RiskLevel(Enum):
    """Enum for risk levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for different trading decisions."""

    # Entry thresholds
    base_entry_threshold: float = 0.7
    volatility_modulated_entry: bool = True
    volatility_multiplier: float = 0.5
    volatility_zscore_threshold: float = 1.0

    # Analyst vs Tactician thresholds
    analyst_confidence_threshold: float = 0.7
    tactician_confidence_threshold: float = 0.8

    # Position management thresholds
    position_scale_up_threshold: float = 0.85
    position_scale_down_threshold: float = 0.6
    position_close_threshold: float = 0.3

    # ML target update thresholds
    ml_target_update_threshold: float = 0.5
    emergency_update_threshold: float = 0.02

    # Ensemble thresholds
    ensemble_agreement_threshold: float = 0.8
    ensemble_minimum_models: int = 3

    # Position closing thresholds
    neutral_signal_threshold: float = 0.5
    tactician_close_threshold: float = 0.6

    # Model performance thresholds
    model_performance_threshold: float = 0.6
    model_degradation_threshold: float = 0.4
    model_retrain_threshold: float = 0.3

    # Regime-specific thresholds
    bull_trend_threshold: float = 0.65
    bear_trend_threshold: float = 0.75
    sideways_threshold: float = 0.8
    sr_zone_threshold: float = 0.7
    high_impact_candle_threshold: float = 0.9


@dataclass
class VolatilityParameters:
    """Volatility-based parameters for position sizing and risk management."""

    # Volatility targeting
    target_volatility: float = 0.15
    volatility_lookback_period: int = 20
    volatility_multiplier: float = 1.0

    # Volatility thresholds
    low_volatility_threshold: float = 0.02
    medium_volatility_threshold: float = 0.05
    high_volatility_threshold: float = 0.10

    # Volatility-based position sizing
    low_volatility_multiplier: float = 1.2
    medium_volatility_multiplier: float = 1.0
    high_volatility_multiplier: float = 0.7

    # Volatility-based stop losses
    volatility_stop_loss_multiplier: float = 2.0
    volatility_take_profit_multiplier: float = 3.0


@dataclass
class ProfitTakingParameters:
    """Multi-stage profit taking parameters."""

    # Multi-stage profit taking
    enable_multi_stage_profit_taking: bool = True
    profit_taking_stages: int = 3

    # Stage-specific targets (ATR multipliers)
    pt1_target_atr_multiplier: float = 1.5
    pt2_target_atr_multiplier: float = 2.5
    pt3_target_atr_multiplier: float = 4.0

    # Stage-specific position sizes
    pt1_position_size_pct: float = 0.33
    pt2_position_size_pct: float = 0.33
    pt3_position_size_pct: float = 0.34

    # Dynamic profit taking
    enable_dynamic_profit_taking: bool = True
    momentum_based_tp: bool = True
    volatility_based_tp: bool = True
    regime_based_tp: bool = True

    # Profit taking confidence adjustments
    pt_confidence_decrease: float = 0.1
    pt_short_term_decrease: float = 0.08


@dataclass
class StopLossParameters:
    """Stop loss and risk management parameters."""

    # ATR-based stop losses
    stop_loss_atr_multiplier: float = 2.0
    trailing_stop_atr_multiplier: float = 1.5

    # Confidence-based stop losses
    stop_loss_confidence_threshold: float = 0.3
    stop_loss_short_term_threshold: float = 0.24
    stop_loss_price_threshold: float = -0.05

    # Dynamic stop losses
    enable_dynamic_stop_loss: bool = True
    volatility_based_sl: bool = True
    regime_based_sl: bool = True

    # Stop loss adjustments
    sl_tightening_threshold: float = 0.4
    sl_loosening_threshold: float = 0.8


@dataclass
class PositionSizingParameters:
    """Position sizing and leverage parameters."""

    # Base position sizing
    base_position_size: float = 0.05
    max_position_size: float = 0.3
    min_position_size: float = 0.01

    # Kelly criterion parameters
    kelly_multiplier: float = 0.25
    fractional_kelly: bool = True

    # Confidence-based scaling
    confidence_based_scaling: bool = True
    low_confidence_multiplier: float = 0.5
    medium_confidence_multiplier: float = 1.0
    high_confidence_multiplier: float = 1.5
    very_high_confidence_multiplier: float = 2.0

    # Successive position parameters
    enable_successive_positions: bool = True
    min_confidence_for_successive: float = 0.85
    max_successive_positions: int = 3
    position_spacing_minutes: int = 15
    size_reduction_factor: float = 0.8
    max_total_exposure: float = 0.3

    # Leverage parameters
    max_leverage: float = 100.0
    min_leverage: float = 10.0
    leverage_confidence_threshold: float = 0.7
    risk_tolerance: float = 0.3


@dataclass
class CooldownParameters:
    """Trade cooldown and timing parameters."""

    # Trade cooldown periods (minutes)
    base_cooldown_minutes: int = 30
    high_confidence_cooldown: int = 15
    low_confidence_cooldown: int = 60

    # Regime-based cooldowns
    bull_trend_cooldown: int = 20
    bear_trend_cooldown: int = 45
    sideways_cooldown: int = 60
    high_impact_cooldown: int = 90

    # Loss-based cooldowns
    loss_cooldown_multiplier: float = 2.0
    consecutive_loss_cooldown: int = 120

    # Volatility-based cooldowns
    high_volatility_cooldown_multiplier: float = 1.5
    low_volatility_cooldown_multiplier: float = 0.8


@dataclass
class DrawdownParameters:
    """Drawdown-based de-risking parameters."""

    # Drawdown thresholds
    warning_drawdown_threshold: float = 0.1
    reduction_drawdown_threshold: float = 0.2
    aggressive_drawdown_threshold: float = 0.3
    emergency_drawdown_threshold: float = 0.4

    # Size reduction factors
    warning_size_reduction: float = 0.9
    reduction_size_reduction: float = 0.7
    aggressive_size_reduction: float = 0.5
    emergency_size_reduction: float = 0.2

    # Daily loss thresholds
    warning_daily_loss: float = 0.05
    reduction_daily_loss: float = 0.08
    emergency_daily_loss: float = 0.10

    # Daily loss size reductions
    warning_daily_reduction: float = 0.8
    reduction_daily_reduction: float = 0.5
    emergency_daily_reduction: float = 0.2


@dataclass
class EnsembleParameters:
    """Ensemble gathering and combination parameters."""

    # Ensemble method
    ensemble_method: EnsembleMethod = EnsembleMethod.CONFIDENCE_WEIGHTED

    # Threshold-based ensemble
    all_threshold_confidence: float = 0.8
    majority_vote_threshold: float = 0.6

    # Weighted ensemble
    analyst_weight: float = 0.4
    tactician_weight: float = 0.3
    strategist_weight: float = 0.3

    # Meta-learner parameters
    meta_learner_type: str = "lightgbm"
    meta_learner_learning_rate: float = 0.1
    meta_learner_n_estimators: int = 100

    # Regime-specific ensemble
    regime_specific_weights: dict[str, float] = None

    # Ensemble validation
    min_ensemble_agreement: float = 0.7
    max_ensemble_disagreement: float = 0.3

    def __post_init__(self):
        if self.regime_specific_weights is None:
            self.regime_specific_weights = {
                "BULL_TREND": 1.2,
                "BEAR_TREND": 0.8,
                "SIDEWAYS_RANGE": 0.9,
                "HIGH_IMPACT_CANDLE": 0.6,
                "SR_ZONE_ACTION": 1.1,
            }


@dataclass
class RiskManagementParameters:
    """Comprehensive risk management parameters."""

    # Portfolio-level risk
    max_portfolio_risk: float = 0.15
    max_correlation_exposure: float = 0.2
    max_sector_exposure: float = 0.3

    # Position-level risk
    max_single_position: float = 0.15
    max_total_exposure: float = 0.3
    max_leverage: float = 10.0

    # Risk metrics
    var_confidence_level: float = 0.95
    max_var_threshold: float = 0.02
    max_cvar_threshold: float = 0.03

    # Dynamic risk adjustment
    enable_dynamic_risk: bool = True
    volatility_scaling: bool = True
    regime_based_risk: bool = True

    # Risk limits
    max_drawdown: float = 0.25
    max_daily_loss: float = 0.1
    max_consecutive_losses: int = 5


@dataclass
class MarketRegimeParameters:
    """Market regime detection and adaptation parameters."""

    # Regime detection
    regime_lookback_period: int = 50
    regime_volatility_threshold: float = 0.02
    regime_trend_threshold: float = 0.01
    regime_stability_threshold: float = 0.7

    # Regime-specific parameters
    bull_trend_multiplier: float = 1.2
    bear_trend_multiplier: float = 0.8
    sideways_multiplier: float = 0.9
    high_impact_multiplier: float = 0.6
    sr_zone_multiplier: float = 1.1

    # Regime transition
    regime_transition_threshold: float = 0.6
    regime_confirmation_periods: int = 3

    # Regime-based optimization
    enable_regime_specific_optimization: bool = True
    regime_specific_constraints: dict[str, dict[str, list[float]]] = None

    def __post_init__(self):
        if self.regime_specific_constraints is None:
            self.regime_specific_constraints = {
                "bull": {
                    "tp_multiplier_range": [2.5, 5.0],
                    "sl_multiplier_range": [1.2, 2.5],
                    "position_size_range": [0.10, 0.25],
                },
                "bear": {
                    "tp_multiplier_range": [2.0, 4.5],
                    "sl_multiplier_range": [1.0, 2.2],
                    "position_size_range": [0.08, 0.20],
                },
                "sideways": {
                    "tp_multiplier_range": [1.5, 3.0],
                    "sl_multiplier_range": [0.8, 1.5],
                    "position_size_range": [0.05, 0.15],
                },
                "sr": {
                    "tp_multiplier_range": [1.8, 3.5],
                    "sl_multiplier_range": [0.9, 1.8],
                    "position_size_range": [0.06, 0.18],
                },
                "candle": {
                    "tp_multiplier_range": [1.2, 2.5],
                    "sl_multiplier_range": [0.6, 1.2],
                    "position_size_range": [0.03, 0.12],
                },
            }


@dataclass
class OptimizationParameters:
    """Hyperparameter optimization parameters."""

    # Optuna parameters
    n_trials: int = 500
    timeout_seconds: int = 3600
    n_jobs: int = -1

    # Optimization objectives
    primary_objective: str = "sharpe_ratio"
    secondary_objectives: list[str] = None

    # Optimization constraints
    min_win_rate: float = 0.4
    max_drawdown_threshold: float = 0.25
    min_profit_factor: float = 1.2

    # Performance optimization parameters
    min_trades_for_optimization: int = 10
    optimization_interval: int = 3600  # 1 hour
    performance_degradation_threshold: float = 0.1

    # Search spaces
    confidence_threshold_range: list[float] = None
    volatility_multiplier_range: list[float] = None
    atr_multiplier_range: list[float] = None
    position_size_range: list[float] = None

    def __post_init__(self):
        if self.secondary_objectives is None:
            self.secondary_objectives = ["win_rate", "profit_factor"]
        if self.confidence_threshold_range is None:
            self.confidence_threshold_range = [0.6, 0.95]
        if self.volatility_multiplier_range is None:
            self.volatility_multiplier_range = [0.5, 2.0]
        if self.atr_multiplier_range is None:
            self.atr_multiplier_range = [1.0, 5.0]
        if self.position_size_range is None:
            self.position_size_range = [0.01, 0.5]


@dataclass
class TimingParameters:
    """Timing and interval parameters for various system components."""

    # Update intervals
    ml_target_update_interval: int = 30  # seconds
    position_monitoring_interval: int = 10  # seconds
    sentinel_monitoring_interval: int = 60  # seconds
    metrics_dashboard_interval: int = 5  # seconds
    performance_optimizer_interval: int = 3600  # 1 hour

    # Training intervals
    enhanced_training_interval: int = 3600  # 1 hour
    retrain_interval_hours: int = 24  # hours
    model_validation_interval: int = 1800  # 30 minutes

    # Cache durations
    feature_cache_duration_minutes: int = 5
    regime_cache_duration_minutes: int = 15
    optimization_cache_duration_minutes: int = 60

    # Data splits
    training_split: float = 0.8
    validation_split: float = 0.15
    test_split: float = 0.05

    # Calibration parameters
    calibration_window: int = 1000
    calibration_interval_hours: int = 6


@dataclass
class ModelTrainingParameters:
    """Model training and validation parameters."""

    # Training configuration
    enable_advanced_model_training: bool = True
    enable_ensemble_training: bool = True
    enable_multi_timeframe_training: bool = True
    enable_adaptive_training: bool = True
    enable_regime_specific_training: bool = True
    enable_dual_model_training: bool = True
    enable_confidence_calibration: bool = True

    # Model performance thresholds
    min_model_accuracy: float = 0.6
    min_model_precision: float = 0.55
    min_model_recall: float = 0.5
    model_degradation_threshold: float = 0.1
    model_retrain_threshold: float = 0.2

    # Training constraints
    max_training_time_hours: int = 2
    min_training_samples: int = 1000
    max_training_samples: int = 50000
    early_stopping_patience: int = 20

    # Validation parameters
    cross_validation_folds: int = 5
    walk_forward_windows: int = 10
    monte_carlo_simulations: int = 100


@dataclass
class MonitoringParameters:
    """System monitoring and performance tracking parameters."""

    # Performance monitoring
    performance_history_size: int = 1000
    model_performance_history_size: int = 100
    monitoring_interval_seconds: int = 60

    # Alert thresholds
    performance_alert_threshold: float = 0.1
    model_degradation_alert_threshold: float = 0.15
    system_health_alert_threshold: float = 0.8

    # Metrics tracking
    enable_detailed_metrics: bool = True
    enable_model_performance_tracking: bool = True
    enable_system_health_monitoring: bool = True

    # Reporting intervals
    daily_report_interval_hours: int = 24
    weekly_report_interval_days: int = 7
    monthly_report_interval_days: int = 30


@dataclass
class FeatureEngineeringParameters:
    """Feature engineering and data processing parameters."""

    # Feature selection
    feature_selection_threshold: float = 0.01
    correlation_threshold: float = 0.95
    pca_variance_threshold: float = 0.95
    min_features: int = 10
    max_features: int = 100

    # Technical indicators
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14

    # Advanced features
    enable_divergence_detection: bool = True
    enable_pattern_recognition: bool = True
    enable_volume_profile: bool = True
    enable_market_microstructure: bool = True
    enable_volatility_targeting: bool = True

    # Feature caching
    enable_feature_caching: bool = True
    feature_cache_size: int = 1000
    feature_cache_ttl_minutes: int = 60


@dataclass
class DualModelSystemParameters:
    """Dual model system configuration parameters."""

    # Analyst model configuration (IF decisions) - multi-timeframe
    analyst_timeframes: list[str] = None
    analyst_confidence_threshold: float = 0.5
    analyst_model_types: list[str] = None

    # Tactician model configuration (WHEN decisions) - 1m timeframe
    tactician_timeframes: list[str] = None
    tactician_confidence_threshold: float = 0.6
    tactician_model_types: list[str] = None

    # Signal management
    enter_signal_validity_duration: int = 120  # 2 minutes in seconds
    signal_check_interval: int = 10  # 10 seconds

    # Confidence thresholds for signals
    neutral_signal_threshold: float = (
        0.5  # NEUTRAL signal when confidence drops below 0.5
    )
    close_signal_threshold: float = 0.4  # CLOSE signal when confidence drops below 0.4

    # Position management thresholds
    position_close_confidence_threshold: float = (
        0.6  # Close positions when tactician confidence drops below 0.6
    )

    # Ensemble analysis
    enable_ensemble_analysis: bool = True
    ensemble_agreement_threshold: float = 0.7
    ensemble_minimum_models: int = 3

    # ML Confidence Predictor integration
    enhanced_training_integration: bool = True
    model_path: str = "models/ml_confidence_predictor"
    min_samples_for_training: int = 1000
    confidence_threshold: float = 0.6
    max_prediction_horizon: int = 1

    # Meta-labeling integration
    enable_meta_labeling: bool = True
    meta_labeling_config: dict[str, Any] = None

    # Dual confidence formula parameters
    final_confidence_minimum: float = 0.216  # 0.5 * 0.6^2
    normalized_confidence_range: float = 0.784  # 1.0 - 0.216
    max_size_multiplier: float = 3.0

    # Kelly criterion parameters
    fractional_kelly_pct: float = 0.75  # 75% fractional Kelly for safety
    historical_trades_for_kelly: int = 200
    default_win_rate: float = 0.5  # Default for less than 200 trades

    # Leverage parameters
    min_leverage: float = 10.0
    max_leverage: float = 100.0
    leverage_confidence_threshold: float = 0.7

    # Position closing parameters
    atr_exit_multiplier: float = 1.5  # Exit if price reverses by 1.5x ATR
    hard_stop_loss_before_liquidation: float = 0.1  # 10% before liquidation
    time_based_exit_hours: int = 2  # 2-hour exit rule

    # Position division strategy
    enable_confidence_increase_check: bool = True
    min_confidence_increase: float = 0.05

    def __post_init__(self):
        if self.analyst_timeframes is None:
            self.analyst_timeframes = ["30m", "15m", "5m"]
        if self.tactician_timeframes is None:
            self.tactician_timeframes = ["1m"]
        if self.analyst_model_types is None:
            self.analyst_model_types = ["tcn", "tabnet", "transformer"]
        if self.tactician_model_types is None:
            self.tactician_model_types = ["lstm", "gru", "transformer"]
        if self.meta_labeling_config is None:
            self.meta_labeling_config = {
                "enable_analyst_labels": True,
                "enable_tactician_labels": True,
                "pattern_detection": {
                    "volatility_threshold": 0.02,
                    "momentum_threshold": 0.01,
                    "volume_threshold": 1.5,
                },
                "entry_prediction": {
                    "prediction_horizon": 5,
                    "max_adverse_excursion": 0.02,
                },
            }


@dataclass
class MetaLabelingParameters:
    """Meta-labeling system configuration parameters."""

    # Enable/disable features
    enable_analyst_labels: bool = True
    enable_tactician_labels: bool = True

    # Pattern detection parameters
    volatility_threshold: float = 0.02
    momentum_threshold: float = 0.01
    volume_threshold: float = 1.5

    # Entry prediction parameters
    prediction_horizon: int = 5  # minutes
    max_adverse_excursion: float = 0.02

    # Analyst label types
    enable_trend_continuation: bool = True
    enable_exhaustion_reversal: bool = True
    enable_range_mean_reversion: bool = True
    enable_breakout_patterns: bool = True
    enable_volatility_patterns: bool = True
    enable_chart_patterns: bool = True
    enable_momentum_patterns: bool = True

    # Tactician label types
    enable_entry_signals: bool = True
    enable_price_extremes: bool = True
    enable_order_returns: bool = True
    enable_adverse_excursion: bool = True
    enable_abort_signals: bool = True

    # Pattern detection thresholds
    trend_strength_threshold: float = 0.6
    reversal_confidence_threshold: float = 0.7
    breakout_confidence_threshold: float = 0.8
    volatility_confidence_threshold: float = 0.6
    momentum_confidence_threshold: float = 0.7

    # Entry signal thresholds
    vwap_reversion_threshold: float = 0.01
    market_order_momentum_threshold: float = 0.02
    micro_breakout_threshold: float = 0.001
    order_imbalance_threshold: float = 0.3
    taker_spike_threshold: float = 3.0


@dataclass
class FeatureEngineeringParameters:
    """Feature engineering system configuration parameters."""

    # Enable/disable features
    enable_advanced_features: bool = True
    enable_multi_timeframe_features: bool = True
    enable_autoencoder_features: bool = True
    enable_legacy_features: bool = True

    # Feature management
    feature_cache_duration: int = 300  # 5 minutes
    enable_feature_selection: bool = True
    max_features: int = 500
    feature_selection_method: str = "mutual_info"

    # Multi-timeframe feature engineering
    enable_mtf_features: bool = True
    enable_timeframe_adaptation: bool = True

    # Advanced feature engineering
    enable_candlestick_patterns: bool = True
    enable_microstructure_features: bool = True
    enable_adaptive_indicators: bool = True
    enable_volatility_regime_modeling: bool = True
    enable_correlation_analysis: bool = True
    enable_momentum_analysis: bool = True
    enable_liquidity_analysis: bool = True

    # Autoencoder feature generation
    autoencoder_hidden_dim: int = 64
    autoencoder_latent_dim: int = 16
    autoencoder_learning_rate: float = 0.001
    autoencoder_epochs: int = 100
    autoencoder_batch_size: int = 32

    # Feature preprocessing
    enable_feature_scaling: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"
    enable_feature_normalization: bool = True
    enable_outlier_detection: bool = True
    outlier_detection_method: str = "isolation_forest"

    # Technical indicators
    enable_rsi: bool = True
    enable_macd: bool = True
    enable_bollinger_bands: bool = True
    enable_stochastic: bool = True
    enable_atr: bool = True
    enable_adx: bool = True
    enable_cci: bool = True

    # Volume indicators
    enable_volume_ma: bool = True
    enable_volume_ratio: bool = True
    enable_obv: bool = True
    enable_vwap: bool = True
    enable_money_flow_index: bool = True

    # Volatility indicators
    enable_historical_volatility: bool = True
    enable_parkinson_volatility: bool = True
    enable_garman_klass_volatility: bool = True
    enable_rogers_satchell_volatility: bool = True

    # Momentum indicators
    enable_momentum: bool = True
    enable_rate_of_change: bool = True
    enable_williams_r: bool = True
    enable_ultimate_oscillator: bool = True

    # Timeframe-specific parameters
    execution_timeframe: str = "1m"
    tactical_timeframe: str = "15m"
    strategic_timeframe: str = "1h"

    # Feature selection thresholds
    mutual_info_threshold: float = 0.01
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01


@dataclass
class OrderManagementParameters:
    """Order management and execution parameters."""

    # Enhanced order manager
    enable_enhanced_order_manager: bool = True
    enable_async_order_executor: bool = True
    enable_chase_micro_breakout: bool = True
    enable_limit_order_return: bool = True
    enable_partial_fill_management: bool = True

    # Order execution parameters
    max_order_retries: int = 3
    order_timeout_seconds: int = 30
    slippage_tolerance: float = 0.001
    volume_threshold: float = 1.5
    momentum_threshold: float = 0.02

    # Execution strategies
    enable_immediate_execution: bool = True
    enable_batch_execution: bool = True
    enable_twap_execution: bool = True
    enable_vwap_execution: bool = True
    enable_iceberg_execution: bool = True
    enable_adaptive_execution: bool = True

    # Strategy-specific parameters
    immediate_max_slippage: float = 0.001
    immediate_timeout_seconds: int = 30

    batch_size: float = 0.1
    batch_interval: int = 5

    twap_duration_minutes: int = 10
    twap_intervals: int = 20

    vwap_volume_threshold: float = 1.5
    vwap_price_deviation: float = 0.002

    iceberg_qty: float = 0.1
    iceberg_display_qty: float = 0.01

    adaptive_dynamic_slippage: bool = True
    adaptive_market_impact_aware: bool = True

    # Order tracking
    enable_order_tracking: bool = True
    order_tracking_interval: int = 5  # seconds
    max_tracking_duration: int = 3600  # 1 hour

    # Performance monitoring
    enable_execution_analytics: bool = True
    enable_slippage_monitoring: bool = True
    enable_fill_rate_tracking: bool = True

    # Risk management
    max_order_size: float = 0.3  # 30% of portfolio
    max_daily_orders: int = 100
    max_concurrent_orders: int = 10

    # Order validation
    enable_order_validation: bool = True
    min_order_size: float = 0.001
    max_order_size_usdt: float = 5000

    # Error handling
    enable_retry_on_failure: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5


@dataclass
class ModelTrainingParameters:
    """Model training and update parameters."""

    # Training enable/disable
    enable_continuous_training: bool = True
    enable_adaptive_training: bool = True
    enable_incremental_training: bool = True
    enable_full_training: bool = True

    # Training intervals and triggers
    training_interval_hours: int = 24
    min_samples_for_retraining: int = 1000
    performance_degradation_threshold: float = 0.1
    accuracy_degradation_threshold: float = 0.05

    # Model calibration
    enable_model_calibration: bool = True
    enable_confidence_calibration: bool = True
    enable_ensemble_calibration: bool = True
    enable_regime_calibration: bool = True

    # Training strategies
    enable_ensemble_training: bool = True
    enable_regime_specific_training: bool = True
    enable_multi_timeframe_training: bool = True
    enable_dual_model_training: bool = True

    # Training parameters
    batch_size: int = 1000
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # Adaptive training
    dynamic_learning_rate: bool = True
    performance_threshold: float = 0.7
    adaptive_batch_size: bool = True
    adaptive_epochs: bool = True

    # Incremental training
    update_frequency: int = 100
    memory_size: int = 10000
    incremental_learning_rate: float = 0.0001
    forgetting_factor: float = 0.9

    # Model types
    analyst_model_types: list[str] = None
    tactician_model_types: list[str] = None
    ensemble_methods: list[str] = None

    # Training data
    min_training_samples: int = 5000
    max_training_samples: int = 100000
    data_quality_threshold: float = 0.8
    feature_importance_threshold: float = 0.01

    # Validation
    cross_validation_folds: int = 5
    walk_forward_windows: int = 10
    monte_carlo_simulations: int = 100

    # Performance monitoring
    enable_performance_tracking: bool = True
    performance_history_size: int = 100
    model_comparison_metrics: list[str] = None

    # Model storage
    enable_model_versioning: bool = True
    max_model_versions: int = 10
    model_backup_enabled: bool = True

    # Training optimization
    enable_hyperparameter_optimization: bool = True
    optimization_trials: int = 100
    optimization_timeout_hours: int = 2

    def __post_init__(self):
        if self.analyst_model_types is None:
            self.analyst_model_types = ["tcn", "tabnet", "transformer"]
        if self.tactician_model_types is None:
            self.tactician_model_types = ["lstm", "gru", "transformer"]
        if self.ensemble_methods is None:
            self.ensemble_methods = ["voting", "stacking", "bagging", "boosting"]
        if self.model_comparison_metrics is None:
            self.model_comparison_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
            ]


# Main Optuna Configuration
OPTUNA_CONFIG = {
    "confidence_thresholds": ConfidenceThresholds(),
    "volatility_parameters": VolatilityParameters(),
    "profit_taking_parameters": ProfitTakingParameters(),
    "stop_loss_parameters": StopLossParameters(),
    "position_sizing_parameters": PositionSizingParameters(),
    "cooldown_parameters": CooldownParameters(),
    "drawdown_parameters": DrawdownParameters(),
    "ensemble_parameters": EnsembleParameters(),
    "risk_management_parameters": RiskManagementParameters(),
    "market_regime_parameters": MarketRegimeParameters(),
    "optimization_parameters": OptimizationParameters(),
    "timing_parameters": TimingParameters(),
    "model_training_parameters": ModelTrainingParameters(),
    "monitoring_parameters": MonitoringParameters(),
    "feature_engineering_parameters": FeatureEngineeringParameters(),
    "dual_model_system_parameters": DualModelSystemParameters(),
    "meta_labeling_parameters": MetaLabelingParameters(),
    "order_management_parameters": OrderManagementParameters(),
}


def get_optuna_config() -> dict[str, Any]:
    """Get the complete Optuna configuration."""
    return OPTUNA_CONFIG


def get_parameter_value(parameter_path: str, default: Any = None) -> Any:
    """
    Get a parameter value from the Optuna configuration using dot notation.

    Args:
        parameter_path: Dot-separated path to the parameter (e.g., "confidence_thresholds.base_entry_threshold")
        default: Default value if parameter not found

    Returns:
        Parameter value or default
    """
    try:
        keys = parameter_path.split(".")
        value = OPTUNA_CONFIG

        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            elif hasattr(value, key):
                value = getattr(value, key)
            else:
                return default

        return value
    except (KeyError, AttributeError):
        return default


def update_parameter_value(parameter_path: str, new_value: Any) -> bool:
    """
    Update a parameter value in the Optuna configuration using dot notation.

    Args:
        parameter_path: Dot-separated path to the parameter
        new_value: New value to set

    Returns:
        True if update successful, False otherwise
    """
    try:
        keys = parameter_path.split(".")
        current = OPTUNA_CONFIG

        # Navigate to the parent of the target
        for key in keys[:-1]:
            if isinstance(current, dict):
                current = current[key]
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return False

        # Set the value
        target_key = keys[-1]
        if isinstance(current, dict):
            current[target_key] = new_value
        elif hasattr(current, target_key):
            setattr(current, target_key, new_value)
        else:
            return False

        return True
    except (KeyError, AttributeError):
        return False


def get_optimizable_parameters() -> dict[str, Any]:
    """
    Get all parameters that can be optimized by Optuna.

    Returns:
        Dictionary of parameter names and their current values
    """
    optimizable_params = {}

    # Add all dataclass fields that are numeric
    for section_name, section_config in OPTUNA_CONFIG.items():
        if hasattr(section_config, "__dataclass_fields__"):
            for field_name, field_info in section_config.__dataclass_fields__.items():
                if field_info.type in (float, int):
                    param_path = f"{section_name}.{field_name}"
                    optimizable_params[param_path] = getattr(section_config, field_name)

    return optimizable_params


def validate_optuna_config() -> list[str]:
    """
    Validate the Optuna configuration for consistency.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate confidence thresholds
    ct = OPTUNA_CONFIG["confidence_thresholds"]
    if ct.base_entry_threshold <= 0 or ct.base_entry_threshold >= 1:
        errors.append("base_entry_threshold must be between 0 and 1")

    # Validate position sizing
    ps = OPTUNA_CONFIG["position_sizing_parameters"]
    if ps.max_position_size <= ps.min_position_size:
        errors.append("max_position_size must be greater than min_position_size")

    # Validate drawdown parameters
    dd = OPTUNA_CONFIG["drawdown_parameters"]
    if dd.warning_drawdown_threshold >= dd.reduction_drawdown_threshold:
        errors.append(
            "warning_drawdown_threshold must be less than reduction_drawdown_threshold",
        )

    # Validate volatility parameters
    vp = OPTUNA_CONFIG["volatility_parameters"]
    if vp.low_volatility_threshold >= vp.medium_volatility_threshold:
        errors.append(
            "low_volatility_threshold must be less than medium_volatility_threshold",
        )

    # Validate timing parameters
    tp = OPTUNA_CONFIG["timing_parameters"]
    if tp.training_split + tp.validation_split + tp.test_split != 1.0:
        errors.append("training_split + validation_split + test_split must equal 1.0")

    # Validate model training parameters
    mtp = OPTUNA_CONFIG["model_training_parameters"]
    if mtp.min_model_accuracy < 0 or mtp.min_model_accuracy > 1:
        errors.append("min_model_accuracy must be between 0 and 1")

    # Validate feature engineering parameters
    fe = OPTUNA_CONFIG["feature_engineering_parameters"]
    if fe.min_features > fe.max_features:
        errors.append("min_features must be less than or equal to max_features")

    return errors


# Export for backward compatibility
__all__ = [
    "OPTUNA_CONFIG",
    "get_optuna_config",
    "get_parameter_value",
    "update_parameter_value",
    "get_optimizable_parameters",
    "validate_optuna_config",
    "ConfidenceThresholds",
    "VolatilityParameters",
    "ProfitTakingParameters",
    "StopLossParameters",
    "PositionSizingParameters",
    "CooldownParameters",
    "DrawdownParameters",
    "EnsembleParameters",
    "RiskManagementParameters",
    "MarketRegimeParameters",
    "OptimizationParameters",
    "TimingParameters",
    "ModelTrainingParameters",
    "MonitoringParameters",
    "FeatureEngineeringParameters",
    "EnsembleMethod",
    "RiskLevel",
]
