"""
Multi-Timeframe Configuration for Hybrid NAS-TAS Regime System

This module provides configuration for multi-timeframe trading support,
allowing the system to trade on 1m and 5m timeframes while maintaining
15m regime detection.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeframeType(Enum):
    """Supported timeframe types."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"


class TradingMode(Enum):
    """Trading modes for different timeframes."""
    REGIME_DETECTION = "regime_detection"  # 15m for regime detection
    HIGH_FREQUENCY = "high_frequency"      # 1m for high-frequency trading
    MEDIUM_FREQUENCY = "medium_frequency"  # 5m for medium-frequency trading


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: TimeframeType
    trading_mode: TradingMode
    enabled: bool = True
    weight: float = 1.0
    risk_multiplier: float = 1.0
    signal_threshold: float = 0.6
    confidence_threshold: float = 0.7
    max_position_size: float = 1.0
    stop_loss_threshold: float = 0.02
    take_profit_threshold: float = 0.04
    trading_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe trading."""
    # Primary timeframe for regime detection (always 15m)
    primary_timeframe: TimeframeType = TimeframeType.MINUTE_15
    
    # Trading timeframes
    trading_timeframes: List[TimeframeType] = field(default_factory=lambda: [
        TimeframeType.MINUTE_1,
        TimeframeType.MINUTE_5
    ])
    
    # Timeframe-specific configurations
    timeframe_configs: Dict[TimeframeType, TimeframeConfig] = field(default_factory=dict)
    
    # Cross-timeframe analysis
    enable_cross_timeframe_analysis: bool = True
    correlation_threshold: float = 0.7
    divergence_threshold: float = 0.3
    
    # Risk management
    enable_risk_management: bool = True
    max_total_exposure: float = 1.0
    correlation_penalty: float = 0.1
    
    # Signal aggregation
    signal_aggregation_method: str = "weighted_average"  # "weighted_average", "majority_vote", "consensus"
    timeframe_weights: Dict[TimeframeType, float] = field(default_factory=dict)
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_window: int = 100  # Number of periods to track
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if not self.timeframe_configs:
            self.timeframe_configs = self._create_default_timeframe_configs()
        
        if not self.timeframe_weights:
            self.timeframe_weights = self._create_default_weights()
    
    def _create_default_timeframe_configs(self) -> Dict[TimeframeType, TimeframeConfig]:
        """Create default timeframe configurations."""
        configs = {}
        
        # 15m configuration for regime detection
        configs[TimeframeType.MINUTE_15] = TimeframeConfig(
            timeframe=TimeframeType.MINUTE_15,
            trading_mode=TradingMode.REGIME_DETECTION,
            enabled=True,
            weight=1.0,
            risk_multiplier=1.0,
            signal_threshold=0.7,
            confidence_threshold=0.8,
            max_position_size=1.0,
            stop_loss_threshold=0.03,
            take_profit_threshold=0.06,
            trading_parameters={
                'regime_detection': True,
                'signal_generation': False,
                'position_sizing': False
            }
        )
        
        # 1m configuration for high-frequency trading
        configs[TimeframeType.MINUTE_1] = TimeframeConfig(
            timeframe=TimeframeType.MINUTE_1,
            trading_mode=TradingMode.HIGH_FREQUENCY,
            enabled=True,
            weight=0.3,
            risk_multiplier=0.5,
            signal_threshold=0.8,
            confidence_threshold=0.9,
            max_position_size=0.3,
            stop_loss_threshold=0.01,
            take_profit_threshold=0.02,
            trading_parameters={
                'regime_detection': False,
                'signal_generation': True,
                'position_sizing': True,
                'high_frequency': True,
                'scalping': True
            }
        )
        
        # 5m configuration for medium-frequency trading
        configs[TimeframeType.MINUTE_5] = TimeframeConfig(
            timeframe=TimeframeType.MINUTE_5,
            trading_mode=TradingMode.MEDIUM_FREQUENCY,
            enabled=True,
            weight=0.7,
            risk_multiplier=0.8,
            signal_threshold=0.7,
            confidence_threshold=0.8,
            max_position_size=0.7,
            stop_loss_threshold=0.02,
            take_profit_threshold=0.04,
            trading_parameters={
                'regime_detection': False,
                'signal_generation': True,
                'position_sizing': True,
                'medium_frequency': True,
                'swing_trading': True
            }
        )
        
        return configs
    
    def _create_default_weights(self) -> Dict[TimeframeType, float]:
        """Create default timeframe weights."""
        return {
            TimeframeType.MINUTE_15: 1.0,  # Full weight for regime detection
            TimeframeType.MINUTE_5: 0.7,   # Higher weight for 5m trading
            TimeframeType.MINUTE_1: 0.3    # Lower weight for 1m trading
        }
    
    def get_timeframe_config(self, timeframe: TimeframeType) -> Optional[TimeframeConfig]:
        """Get configuration for a specific timeframe."""
        return self.timeframe_configs.get(timeframe)
    
    def is_timeframe_enabled(self, timeframe: TimeframeType) -> bool:
        """Check if a timeframe is enabled."""
        config = self.get_timeframe_config(timeframe)
        return config.enabled if config else False
    
    def get_timeframe_weight(self, timeframe: TimeframeType) -> float:
        """Get weight for a specific timeframe."""
        return self.timeframe_weights.get(timeframe, 0.0)
    
    def get_trading_timeframes(self) -> List[TimeframeType]:
        """Get list of enabled trading timeframes."""
        return [tf for tf in self.trading_timeframes if self.is_timeframe_enabled(tf)]
    
    def get_regime_detection_timeframe(self) -> TimeframeType:
        """Get the timeframe used for regime detection."""
        return self.primary_timeframe
    
    def validate_configuration(self) -> bool:
        """Validate the multi-timeframe configuration."""
        try:
            # Check that primary timeframe is configured
            if not self.get_timeframe_config(self.primary_timeframe):
                logger.error(f"Primary timeframe {self.primary_timeframe.value} not configured")
                return False
            
            # Check that at least one trading timeframe is enabled
            trading_timeframes = self.get_trading_timeframes()
            if not trading_timeframes:
                logger.error("No trading timeframes enabled")
                return False
            
            # Check that weights sum to reasonable value
            total_weight = sum(self.timeframe_weights.values())
            if total_weight <= 0:
                logger.error("Total timeframe weights must be positive")
                return False
            
            # Check that risk multipliers are positive
            for config in self.timeframe_configs.values():
                if config.risk_multiplier <= 0:
                    logger.error(f"Risk multiplier for {config.timeframe.value} must be positive")
                    return False
            
            logger.info("✅ Multi-timeframe configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


@dataclass
class CrossTimeframeAnalysisConfig:
    """Configuration for cross-timeframe analysis."""
    enable_correlation_analysis: bool = True
    enable_divergence_detection: bool = True
    enable_signal_consensus: bool = True
    enable_risk_aggregation: bool = True
    
    # Correlation analysis
    correlation_window: int = 50
    correlation_threshold: float = 0.7
    correlation_method: str = "pearson"  # "pearson", "spearman", "kendall"
    
    # Divergence detection
    divergence_threshold: float = 0.3
    divergence_window: int = 20
    divergence_method: str = "statistical"  # "statistical", "momentum", "regime"
    
    # Signal consensus
    consensus_threshold: float = 0.6
    consensus_method: str = "weighted_vote"  # "weighted_vote", "majority_vote", "unanimous"
    
    # Risk aggregation
    risk_aggregation_method: str = "var"  # "var", "cvar", "max_drawdown", "volatility"
    risk_window: int = 100
    confidence_level: float = 0.95


@dataclass
class MultiTimeframeTradingConfig:
    """Complete configuration for multi-timeframe trading."""
    multi_timeframe: MultiTimeframeConfig
    cross_timeframe_analysis: CrossTimeframeAnalysisConfig
    
    # Global settings
    enable_multi_timeframe: bool = True
    enable_adaptive_weights: bool = True
    enable_dynamic_risk_management: bool = True
    
    # Performance settings
    enable_performance_optimization: bool = True
    optimization_window: int = 200
    rebalancing_frequency: int = 50  # Rebalance every N periods
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_level: str = "INFO"
    enable_metrics_collection: bool = True
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        if not self.multi_timeframe.validate_configuration():
            raise ValueError("Invalid multi-timeframe configuration")
        
        logger.info("✅ Multi-timeframe trading configuration initialized")
        logger.info(f"   Primary timeframe: {self.multi_timeframe.primary_timeframe.value}")
        logger.info(f"   Trading timeframes: {[tf.value for tf in self.multi_timeframe.get_trading_timeframes()]}")
        logger.info(f"   Cross-timeframe analysis: {'✅ Enabled' if self.cross_timeframe_analysis.enable_correlation_analysis else '❌ Disabled'}")


def create_default_multi_timeframe_config() -> MultiTimeframeTradingConfig:
    """Create default multi-timeframe trading configuration."""
    multi_timeframe = MultiTimeframeConfig()
    cross_timeframe_analysis = CrossTimeframeAnalysisConfig()
    
    return MultiTimeframeTradingConfig(
        multi_timeframe=multi_timeframe,
        cross_timeframe_analysis=cross_timeframe_analysis
    )


def create_high_frequency_config() -> MultiTimeframeTradingConfig:
    """Create configuration optimized for high-frequency trading."""
    multi_timeframe = MultiTimeframeConfig()
    
    # Adjust for high-frequency trading
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_1].weight = 0.8
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_1].max_position_size = 0.5
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_1].signal_threshold = 0.9
    
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].weight = 0.2
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].max_position_size = 0.2
    
    cross_timeframe_analysis = CrossTimeframeAnalysisConfig()
    cross_timeframe_analysis.correlation_window = 20
    cross_timeframe_analysis.consensus_threshold = 0.8
    
    return MultiTimeframeTradingConfig(
        multi_timeframe=multi_timeframe,
        cross_timeframe_analysis=cross_timeframe_analysis
    )


def create_swing_trading_config() -> MultiTimeframeTradingConfig:
    """Create configuration optimized for swing trading."""
    multi_timeframe = MultiTimeframeConfig()
    
    # Adjust for swing trading
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_1].enabled = False
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_1].weight = 0.0
    
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].weight = 1.0
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].max_position_size = 1.0
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].signal_threshold = 0.6
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].stop_loss_threshold = 0.03
    multi_timeframe.timeframe_configs[TimeframeType.MINUTE_5].take_profit_threshold = 0.06
    
    cross_timeframe_analysis = CrossTimeframeAnalysisConfig()
    cross_timeframe_analysis.correlation_window = 100
    cross_timeframe_analysis.consensus_threshold = 0.5
    
    return MultiTimeframeTradingConfig(
        multi_timeframe=multi_timeframe,
        cross_timeframe_analysis=cross_timeframe_analysis
    )


def create_balanced_config() -> MultiTimeframeTradingConfig:
    """Create balanced configuration for mixed trading strategies."""
    return create_default_multi_timeframe_config()