"""Configuration management for the Strategist module using Pydantic."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class StrategyType(str, Enum):
    """Enumeration of available strategy types."""

    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    HYBRID = "hybrid"
    ML_DRIVEN = "ml_driven"


class RiskLevel(str, Enum):
    """Risk level classifications."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TechnicalIndicatorThresholds(BaseModel):
    """Technical indicator threshold configuration."""

    rsi_oversold: float = Field(
        default=30.0, ge=0, le=100, description="RSI oversold threshold"
    )
    rsi_overbought: float = Field(
        default=70.0, ge=0, le=100, description="RSI overbought threshold"
    )
    sma_fast_window: int = Field(default=20, ge=1, description="Fast SMA window period")
    sma_slow_window: int = Field(default=50, ge=1, description="Slow SMA window period")
    volume_ratio_high: float = Field(
        default=1.5, gt=0, description="High volume ratio threshold"
    )
    volume_ratio_low: float = Field(
        default=0.5, gt=0, description="Low volume ratio threshold"
    )
    price_volatility_window: int = Field(
        default=20, ge=1, description="Price volatility calculation window"
    )

    @validator("sma_slow_window")
    def validate_sma_windows(cls, v, values):
        """Ensure slow SMA window is greater than fast SMA window."""
        if "sma_fast_window" in values and v <= values["sma_fast_window"]:
            raise ValueError("Slow SMA window must be greater than fast SMA window")
        return v


class StrategistConfig(BaseModel):
    """Complete configuration for the Strategist component."""

    strategy_interval: int = Field(
        default=1800, ge=60, description="Strategy update interval in seconds"
    )
    max_strategy_history: int = Field(
        default=50, ge=1, description="Maximum strategy history entries to keep"
    )
    enable_risk_management: bool = Field(
        default=True, description="Enable risk management features"
    )
    min_confidence_threshold: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Minimum confidence threshold for strategies",
    )
    strategy_type: StrategyType = Field(
        default=StrategyType.TECHNICAL_ANALYSIS, description="Type of strategy to use"
    )
    technical_indicator_thresholds: TechnicalIndicatorThresholds = Field(
        default_factory=TechnicalIndicatorThresholds,
        description="Technical indicator thresholds",
    )

    # Performance optimization settings
    cache_ttl: int = Field(
        default=300, ge=0, description="Cache time-to-live in seconds"
    )
    use_vectorized_calculations: bool = Field(
        default=True, description="Use vectorized calculations for performance"
    )
    parallel_indicator_calculation: bool = Field(
        default=True, description="Calculate indicators in parallel"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            Enum: lambda v: v.value,
        }


class MarketIndicators(BaseModel):
    """Market indicator data structure."""

    rsi: Optional[float] = Field(
        None, ge=0, le=100, description="Relative Strength Index"
    )
    sma_fast: Optional[float] = Field(None, description="Fast Simple Moving Average")
    sma_slow: Optional[float] = Field(None, description="Slow Simple Moving Average")
    volume_ratio: Optional[float] = Field(None, gt=0, description="Volume ratio")
    price_change_percent: Optional[float] = Field(
        None, description="Price change percentage"
    )
    volatility: Optional[float] = Field(None, ge=0, description="Price volatility")
    sma_trend: Optional[str] = Field(None, description="SMA trend direction")


class StrategyResult(BaseModel):
    """Strategy generation result structure."""

    direction: str = Field(..., description="Trading direction: BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=1, description="Strategy confidence level")
    reasoning: list[str] = Field(
        default_factory=list, description="Reasoning for the strategy"
    )
    timestamp: str = Field(..., description="Strategy generation timestamp")

    # Optional fields
    market_health_score: Optional[float] = Field(
        None, ge=0, le=1, description="Market health score"
    )
    liquidation_risk: Optional[str] = Field(None, description="Liquidation risk level")
    dual_model_direction: Optional[str] = Field(
        None, description="Direction from dual model system"
    )
    dual_model_confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Confidence from dual model system"
    )

    # Risk management fields
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    max_position_size: Optional[float] = Field(
        None, description="Maximum position size recommendation"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow additional fields for flexibility
