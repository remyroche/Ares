#!/usr/bin/env python3
"""
Regime and Support/Resistance Tracker

This module provides comprehensive tracking of market regime detection and
support/resistance level identification for monitoring trading performance
and model effectiveness across different market conditions.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd
from scipy import stats
import numpy as np
import os

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


class RegimeType(Enum):
    """Market regime types."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class SRLevelType(Enum):
    """Support/Resistance level types."""

    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"
    FIBONACCI = "fibonacci"
    VOLUME_PROFILE = "volume_profile"
    PSYCHOLOGICAL = "psychological"
    MOVING_AVERAGE = "moving_average"
    TRENDLINE = "trendline"


class RegimeConfidence(Enum):
    """Regime detection confidence levels."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


@dataclass
class RegimeDetection:
    """Market regime detection record."""

    regime_id: str
    timestamp: datetime
    symbol: str
    timeframe: str

    # Regime information
    current_regime: RegimeType
    confidence: float
    duration_minutes: float
    previous_regime: RegimeType | None = None
    regime_transition_time: datetime | None = None

    # Detection signals
    price_action_score: float = 0.0
    volume_score: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    trend_strength_score: float = 0.0

    # Technical indicators
    rsi: float | None = None
    macd_signal: float | None = None
    bollinger_position: float | None = None
    atr_percentile: float | None = None
    volume_ratio: float | None = None

    # Market conditions
    current_price: float = 0.0
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    price_change_24h: float = 0.0
    volume_24h: float = 0.0

    # Regime characteristics
    expected_duration_minutes: float = 0.0
    transition_probability: dict[str, float] = field(default_factory=dict)
    regime_strength: float = 0.0

    # Validation (filled when regime ends)
    actual_duration_minutes: float | None = None
    regime_prediction_accuracy: float | None = None
    ended_as_expected: bool | None = None


@dataclass
class SupportResistanceLevel:
    """Support/Resistance level record."""

    level_id: str
    timestamp: datetime
    symbol: str
    timeframe: str

    # Level details
    level_type: SRLevelType
    price: float
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    # Level characteristics
    touch_count: int = 0
    first_touch_time: datetime | None = None
    last_touch_time: datetime | None = None
    touches_history: list[datetime] = field(default_factory=list)

    # Price action around level
    max_penetration: float = 0.0  # Maximum price beyond level
    avg_rejection_strength: float = 0.0
    volume_at_touches: list[float] = field(default_factory=list)

    # Distance metrics
    distance_from_current: float = 0.0  # Percentage
    distance_from_current_abs: float = 0.0

    # Level status
    is_active: bool = True
    is_broken: bool = False
    break_timestamp: datetime | None = None
    break_volume: float | None = None

    # Predictive metrics
    breakout_probability: float = 0.0
    hold_probability: float = 0.0
    bounce_probability: float = 0.0

    # Performance tracking
    successful_bounces: int = 0
    failed_bounces: int = 0
    success_rate: float = 0.0


@dataclass
class RegimeTransition:
    """Regime transition record."""

    transition_id: str
    timestamp: datetime
    symbol: str
    timeframe: str

    # Transition details
    from_regime: RegimeType
    to_regime: RegimeType
    transition_duration_minutes: float

    # Trigger analysis
    primary_trigger: str  # "price", "volume", "volatility", "external"
    trigger_strength: float
    trigger_details: dict[str, Any] = field(default_factory=dict)

    # Market conditions during transition
    price_at_transition: float
    volume_spike: float  # Volume increase factor
    volatility_spike: float

    # Prediction accuracy
    was_predicted: bool = False
    prediction_accuracy: float | None = None
    early_warning_time_minutes: float | None = None

    # Impact analysis
    price_impact_1h: float | None = None
    price_impact_4h: float | None = None
    volume_impact_1h: float | None = None

    # Model performance during transition
    model_confidence_before: float | None = None
    model_confidence_after: float | None = None
    prediction_errors_during: list[float] = field(default_factory=list)


@dataclass
class TradingPerformanceByRegime:
    """Trading performance analysis by regime."""

    analysis_id: str
    timestamp: datetime
    regime_type: RegimeType
    analysis_period_days: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Performance metrics
    total_pnl: float
    average_pnl_per_trade: float
    max_profit: float
    max_loss: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    average_trade_duration_minutes: float
    sharpe_ratio: float | None = None

    # Model performance in this regime
    prediction_accuracy: float
    avg_confidence: float
    model_effectiveness: dict[str, float] = field(default_factory=dict)

    # Market characteristics during regime
    avg_volatility: float
    avg_volume: float
    price_trend_strength: float

    # Recommendations
    regime_trading_recommendations: list[str] = field(default_factory=list)
    position_sizing_recommendations: dict[str, float] = field(default_factory=dict)


class RegimeSRTracker:
    """
    Comprehensive tracker for market regime detection and support/resistance
    level identification and performance monitoring.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize regime and S/R tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RegimeSRTracker")

        # Configuration
        self.tracker_config = config.get("regime_sr_tracker", {})
        self.enable_regime_tracking = self.tracker_config.get(
            "enable_regime_tracking",
            True,
        )
        self.enable_sr_tracking = self.tracker_config.get("enable_sr_tracking", True)
        self.enable_performance_analysis = self.tracker_config.get(
            "enable_performance_analysis",
            True,
        )
        # Dispatcher integration (Method A)
        self.enable_dispatcher = bool(self.tracker_config.get("enable_dispatcher", True))
        self.dispatcher_manifest_path = self.tracker_config.get("dispatcher_manifest_path", None)
        self.dispatcher_manifest: dict[str, Any] | None = None
        if self.enable_dispatcher and self.dispatcher_manifest_path and os.path.exists(self.dispatcher_manifest_path):
            try:
                with open(self.dispatcher_manifest_path, "r") as jf:
                    self.dispatcher_manifest = json.load(jf)
                self.logger.info(f"Loaded dispatcher manifest: {self.dispatcher_manifest_path}")
            except Exception as _e:
                self.logger.warning(f"Failed to load dispatcher manifest: {_e}")

        # Detection parameters
        self.regime_detection_interval = self.tracker_config.get(
            "regime_detection_interval",
            60,
        )  # seconds
        self.sr_update_interval = self.tracker_config.get(
            "sr_update_interval",
            300,
        )  # seconds
        self.min_regime_duration = self.tracker_config.get(
            "min_regime_duration",
            30,
        )  # minutes
        self.sr_touch_threshold = self.tracker_config.get(
            "sr_touch_threshold",
            0.002,
        )  # 0.2%

        # Storage
        self.regime_detections: dict[str, RegimeDetection] = {}
        self.sr_levels: dict[str, SupportResistanceLevel] = {}
        self.regime_transitions: dict[str, RegimeTransition] = {}
        self.performance_by_regime: dict[str, TradingPerformanceByRegime] = {}

        # Current state
        self.current_regime: dict[
            str,
            RegimeDetection,
        ] = {}  # symbol_timeframe -> detection
        self.active_sr_levels: dict[
            str,
            list[SupportResistanceLevel],
        ] = {}  # symbol_timeframe -> levels

        # Historical data cache
        self.price_history: dict[str, pd.DataFrame] = {}
        self.volume_history: dict[str, pd.DataFrame] = {}

        # Statistics
        self.tracking_stats = {
            "regimes_detected": 0,
            "sr_levels_identified": 0,
            "regime_transitions": 0,
            "performance_analyses": 0,
            "last_update": datetime.now(),
        }

        self.is_initialized = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid regime tracker configuration"),
            AttributeError: (False, "Missing required tracker parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="regime tracker initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the regime and S/R tracker.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Regime and S/R Tracker...")

            # Initialize storage backend
            await self._initialize_storage()

            # Load historical data
            await self._load_historical_data()

            # Initialize detection algorithms
            await self._initialize_detection_algorithms()

            # Start background tasks
            await self._start_background_tasks()

            self.is_initialized = True
            self.logger.info("✅ Regime and S/R Tracker initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Regime and S/R Tracker initialization failed: {e}",
            )
            return False

    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        try:
            storage_backend = self.config.get("monitoring", {}).get(
                "storage_backend",
                "sqlite",
            )

            if storage_backend == "sqlite":
                from src.database.sqlite_manager import SQLiteManager

                self.storage_manager = SQLiteManager(self.config)
                await self.storage_manager.initialize()
                await self._create_regime_sr_tables()

            self.logger.info(
                f"Regime/SR tracker storage backend '{storage_backend}' initialized",
            )

        except Exception as e:
            self.logger.exception(failed("Failed to initialize storage: {e}"))
            raise

    async def _create_regime_sr_tables(self) -> None:
        """Create database tables for regime and S/R tracking."""
        try:
            tables = [
                """
                CREATE TABLE IF NOT EXISTS regime_detections (
                    regime_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    timeframe TEXT,
                    current_regime TEXT,
                    confidence REAL,
                    duration_minutes REAL,
                    price_action_score REAL,
                    volume_score REAL,
                    volatility_score REAL,
                    current_price REAL,
                    regime_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS sr_levels (
                    level_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    timeframe TEXT,
                    level_type TEXT,
                    price REAL,
                    strength REAL,
                    confidence REAL,
                    touch_count INTEGER,
                    is_active BOOLEAN,
                    is_broken BOOLEAN,
                    success_rate REAL,
                    level_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS regime_transitions (
                    transition_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    timeframe TEXT,
                    from_regime TEXT,
                    to_regime TEXT,
                    transition_duration_minutes REAL,
                    primary_trigger TEXT,
                    trigger_strength REAL,
                    was_predicted BOOLEAN,
                    transition_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS regime_performance (
                    analysis_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    regime_type TEXT,
                    analysis_period_days INTEGER,
                    total_trades INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    max_drawdown REAL,
                    prediction_accuracy REAL,
                    performance_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ]

            for table_sql in tables:
                await self.storage_manager.execute_query(table_sql)

            self.logger.info("Regime/SR tracking tables created successfully")

        except Exception as e:
            self.logger.exception(failed("Failed to create regime/SR tables: {e}"))
            raise

    async def _load_historical_data(self) -> None:
        """Load historical regime and S/R data."""
        try:
            # Load recent regime detections
            cutoff_date = datetime.now() - timedelta(days=7)

            if hasattr(self, "storage_manager"):
                # Load recent regime detections
                query = "SELECT * FROM regime_detections WHERE timestamp >= ? ORDER BY timestamp DESC"
                regime_results = await self.storage_manager.execute_query(
                    query,
                    (cutoff_date,),
                )

                # Load active S/R levels
                query = "SELECT * FROM sr_levels WHERE is_active = 1 ORDER BY timestamp DESC"
                sr_results = await self.storage_manager.execute_query(query)

                self.logger.info(
                    f"Loaded {len(regime_results)} regime detections and {len(sr_results)} S/R levels",
                )

        except Exception as e:
            self.logger.exception(failed("Failed to load historical data: {e}"))
            # Non-critical error, continue

    async def _initialize_detection_algorithms(self) -> None:
        """Initialize regime and S/R detection algorithms."""
        try:
            # Initialize regime detection parameters
            self.regime_detection_params = {
                "volatility_window": 20,
                "trend_window": 50,
                "volume_window": 20,
                "momentum_window": 14,
            }

            # Initialize S/R detection parameters
            self.sr_detection_params = {
                "lookback_periods": 100,
                "min_touches": 2,
                "strength_threshold": 0.3,
                "clustering_threshold": 0.005,  # 0.5%
            }

            self.logger.info("Detection algorithms initialized")

        except Exception as e:
            self.logger.exception(failed("Failed to initialize detection algorithms: {e}"))
            raise

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            if self.enable_regime_tracking:
                asyncio.create_task(self._periodic_regime_detection())

            if self.enable_sr_tracking:
                asyncio.create_task(self._periodic_sr_update())

            if self.enable_performance_analysis:
                asyncio.create_task(self._periodic_performance_analysis())

            self.logger.info("Background regime/SR tracking tasks started")

        except Exception as e:
            self.logger.exception(failed("Failed to start background tasks: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="regime detection",
    )
    async def detect_current_regime(
        self,
        symbol: str,
        timeframe: str,
        market_data: pd.DataFrame = None,
    ) -> RegimeDetection | None:
        """
        Detect current market regime for a symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            market_data: Optional market data (if not provided, will fetch)

        Returns:
            RegimeDetection: Current regime detection or None if failed
        """
        try:
            if market_data is None:
                market_data = await self._get_market_data(symbol, timeframe)

            if market_data is None or market_data.empty:
                self.logger.warning(
                    f"No market data available for {symbol} {timeframe}",
                )
                return None

            # Calculate regime indicators
            regime_indicators = await self._calculate_regime_indicators(market_data)

            # Determine regime type
            regime_type, confidence = await self._classify_regime(regime_indicators)

            # Create regime detection
            regime_id = f"{symbol}_{timeframe}_{int(time.time())}"

            detection = RegimeDetection(
                regime_id=regime_id,
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                current_regime=regime_type,
                confidence=confidence,
                duration_minutes=0.0,  # Will be updated as regime continues
                price_action_score=regime_indicators.get("price_action_score", 0.0),
                volume_score=regime_indicators.get("volume_score", 0.0),
                volatility_score=regime_indicators.get("volatility_score", 0.0),
                momentum_score=regime_indicators.get("momentum_score", 0.0),
                trend_strength_score=regime_indicators.get("trend_strength_score", 0.0),
                rsi=regime_indicators.get("rsi"),
                macd_signal=regime_indicators.get("macd_signal"),
                bollinger_position=regime_indicators.get("bollinger_position"),
                atr_percentile=regime_indicators.get("atr_percentile"),
                volume_ratio=regime_indicators.get("volume_ratio"),
                current_price=float(market_data["close"].iloc[-1]),
                price_change_1h=regime_indicators.get("price_change_1h", 0.0),
                price_change_4h=regime_indicators.get("price_change_4h", 0.0),
                price_change_24h=regime_indicators.get("price_change_24h", 0.0),
                volume_24h=regime_indicators.get("volume_24h", 0.0),
                regime_strength=confidence,
                transition_probability=await self._calculate_transition_probabilities(
                    regime_indicators,
                ),
            )

            # Check for regime transition
            key = f"{symbol}_{timeframe}"
            if key in self.current_regime:
                previous_detection = self.current_regime[key]
                if previous_detection.current_regime != regime_type:
                    # Regime transition detected
                    await self._handle_regime_transition(previous_detection, detection)
                else:
                    # Update duration
                    duration = (
                        detection.timestamp - previous_detection.timestamp
                    ).total_seconds() / 60.0
                    detection.duration_minutes = duration

            # Store detection
            self.regime_detections[regime_id] = detection
            self.current_regime[key] = detection

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_regime_detection(detection)

            # Update statistics
            self.tracking_stats["regimes_detected"] += 1
            self.tracking_stats["last_update"] = datetime.now()

            self.logger.debug(
                f"Detected regime {regime_type.value} for {symbol} {timeframe} (confidence: {confidence:.2f})",
            )

            return detection

        except Exception as e:
            self.logger.exception(
                f"Failed to detect regime for {symbol} {timeframe}: {e}",
            )
            return None

    async def _get_market_data(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Get market data for regime analysis."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.price_history:
                data = self.price_history[cache_key]
                # Check if data is recent enough
                if not data.empty and data.index[-1] >= datetime.now() - timedelta(
                    minutes=5,
                ):
                    return data

            # Fetch new data (placeholder - integrate with your data source)
            # This would typically call your data downloader or exchange API
            data = await self._fetch_market_data(symbol, timeframe)

            # Cache the data
            if data is not None and not data.empty:
                self.price_history[cache_key] = data

            return data

        except Exception as e:
            self.logger.exception(
                f"Failed to get market data for {symbol} {timeframe}: {e}",
            )
            return None

    async def _fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Fetch market data from data source."""
        try:
            # Placeholder implementation - integrate with your data source
            # This should call your exchange API or data downloader

            # For now, return None to indicate no data available
            # In a real implementation, you would:
            # 1. Call your exchange API
            # 2. Get the last N candles for this timeframe
            # 3. Return as DataFrame with OHLCV data

            self.logger.debug(f"Fetching market data for {symbol} {timeframe}")
            return None

        except Exception as e:
            self.logger.exception(failed("Failed to fetch market data: {e}"))
            return None

    async def _calculate_regime_indicators(
        self,
        data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate indicators for regime classification."""
        try:
            indicators = {}

            # Price action indicators
            if len(data) >= 20:
                # Trend strength
                returns = data["close"].pct_change()
                indicators["price_change_1h"] = (
                    returns.tail(4).sum() if len(returns) >= 4 else 0.0
                )
                indicators["price_change_4h"] = (
                    returns.tail(16).sum() if len(returns) >= 16 else 0.0
                )
                indicators["price_change_24h"] = (
                    returns.tail(96).sum() if len(returns) >= 96 else 0.0
                )

                # Trend strength score
                ema_short = data["close"].ewm(span=10).mean()
                ema_long = data["close"].ewm(span=30).mean()
                trend_ratio = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[
                    -1
                ]
                indicators["trend_strength_score"] = min(abs(trend_ratio) * 10, 1.0)

                # Price action score
                price_momentum = data["close"].iloc[-1] / data["close"].iloc[-20] - 1
                indicators["price_action_score"] = min(abs(price_momentum) * 5, 1.0)

            # Volume indicators
            if "volume" in data.columns and len(data) >= 20:
                volume_sma = data["volume"].rolling(window=20).mean()
                current_volume_ratio = data["volume"].iloc[-1] / volume_sma.iloc[-1]
                indicators["volume_ratio"] = current_volume_ratio
                indicators["volume_score"] = min(current_volume_ratio / 2.0, 1.0)
                indicators["volume_24h"] = (
                    data["volume"].tail(96).sum() if len(data) >= 96 else 0.0
                )

            # Volatility indicators
            if len(data) >= 20:
                # ATR
                high_low = data["high"] - data["low"]
                high_close = abs(data["high"] - data["close"].shift())
                low_close = abs(data["low"] - data["close"].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                    axis=1,
                )
                atr = true_range.rolling(window=14).mean()

                # Volatility percentile
                if len(atr) >= 50:
                    atr_percentile = (
                        stats.percentileofscore(atr.tail(50), atr.iloc[-1]) / 100.0
                    )
                    indicators["atr_percentile"] = atr_percentile
                    indicators["volatility_score"] = atr_percentile

            # Technical indicators
            if len(data) >= 14:
                # RSI
                delta = data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi"] = rsi.iloc[-1]

            if len(data) >= 26:
                # MACD
                ema12 = data["close"].ewm(span=12).mean()
                ema26 = data["close"].ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                indicators["macd_signal"] = signal.iloc[-1]

            if len(data) >= 20:
                # Bollinger Bands position
                sma20 = data["close"].rolling(window=20).mean()
                std20 = data["close"].rolling(window=20).std()
                bb_upper = sma20 + 2 * std20
                bb_lower = sma20 - 2 * std20
                bb_position = (data["close"].iloc[-1] - bb_lower.iloc[-1]) / (
                    bb_upper.iloc[-1] - bb_lower.iloc[-1]
                )
                indicators["bollinger_position"] = bb_position

            # Momentum indicators
            if len(data) >= 14:
                momentum = data["close"].iloc[-1] / data["close"].iloc[-14] - 1
                indicators["momentum_score"] = min(abs(momentum) * 5, 1.0)

            return indicators

        except Exception as e:
            self.logger.exception(failed("Failed to calculate regime indicators: {e}"))
            return {}

    async def _classify_regime(
        self,
        indicators: dict[str, float],
    ) -> tuple[RegimeType, float]:
        """Classify regime based on indicators."""
        try:
            # Regime classification logic
            scores = {}

            # Bull trend detection
            bull_score = 0.0
            if indicators.get("price_change_24h", 0) > 0.02:  # 2% daily gain
                bull_score += 0.3
            if indicators.get("trend_strength_score", 0) > 0.5:
                bull_score += 0.2
            if indicators.get("rsi", 50) > 60:
                bull_score += 0.2
            if indicators.get("macd_signal", 0) > 0:
                bull_score += 0.3
            scores[RegimeType.BULL_TREND] = bull_score

            # Bear trend detection
            bear_score = 0.0
            if indicators.get("price_change_24h", 0) < -0.02:  # 2% daily loss
                bear_score += 0.3
            if indicators.get("trend_strength_score", 0) > 0.5:
                bear_score += 0.2
            if indicators.get("rsi", 50) < 40:
                bear_score += 0.2
            if indicators.get("macd_signal", 0) < 0:
                bear_score += 0.3
            scores[RegimeType.BEAR_TREND] = bear_score

            # High volatility detection
            high_vol_score = 0.0
            if indicators.get("atr_percentile", 0.5) > 0.8:
                high_vol_score += 0.4
            if indicators.get("volatility_score", 0) > 0.7:
                high_vol_score += 0.3
            if indicators.get("volume_ratio", 1.0) > 2.0:
                high_vol_score += 0.3
            scores[RegimeType.HIGH_VOLATILITY] = high_vol_score

            # Low volatility detection
            low_vol_score = 0.0
            if indicators.get("atr_percentile", 0.5) < 0.2:
                low_vol_score += 0.4
            if indicators.get("volatility_score", 0) < 0.3:
                low_vol_score += 0.3
            if abs(indicators.get("price_change_24h", 0)) < 0.01:
                low_vol_score += 0.3
            scores[RegimeType.LOW_VOLATILITY] = low_vol_score

            # Sideways detection
            sideways_score = 0.0
            if (
                abs(indicators.get("price_change_24h", 0)) < 0.015
            ):  # Less than 1.5% movement
                sideways_score += 0.3
            if indicators.get("trend_strength_score", 0) < 0.3:
                sideways_score += 0.3
            if 40 < indicators.get("rsi", 50) < 60:
                sideways_score += 0.2
            if (
                indicators.get("bollinger_position", 0.5) > 0.3
                and indicators.get("bollinger_position", 0.5) < 0.7
            ):
                sideways_score += 0.2
            scores[RegimeType.SIDEWAYS] = sideways_score

            # Breakout detection
            breakout_score = 0.0
            if indicators.get("volume_ratio", 1.0) > 2.5:
                breakout_score += 0.4
            if abs(indicators.get("price_change_1h", 0)) > 0.005:  # 0.5% hourly move
                breakout_score += 0.3
            bb_pos = indicators.get("bollinger_position", 0.5)
            if bb_pos > 0.9 or bb_pos < 0.1:
                breakout_score += 0.3
            scores[RegimeType.BREAKOUT] = breakout_score

            # Find regime with highest score
            if not scores:
                return RegimeType.SIDEWAYS, 0.5

            best_regime = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(scores[best_regime], 1.0)

            # Minimum confidence threshold
            if confidence < 0.3:
                return RegimeType.SIDEWAYS, confidence

            return best_regime, confidence

        except Exception as e:
            self.logger.exception(failed("Failed to classify regime: {e}"))
            return RegimeType.SIDEWAYS, 0.5

    async def _calculate_transition_probabilities(
        self,
        indicators: dict[str, float],
    ) -> dict[str, float]:
        """Calculate regime transition probabilities."""
        try:
            probabilities = {}

            # Simple transition probability calculation
            # In practice, this would use historical transition data

            for regime in RegimeType:
                if regime == RegimeType.BULL_TREND:
                    prob = 0.1
                    if indicators.get("momentum_score", 0) > 0.7:
                        prob += 0.2
                    if indicators.get("volume_score", 0) > 0.6:
                        prob += 0.1
                elif regime == RegimeType.BEAR_TREND:
                    prob = 0.1
                    if indicators.get("rsi", 50) < 30:
                        prob += 0.2
                    if indicators.get("price_change_4h", 0) < -0.01:
                        prob += 0.1
                elif regime == RegimeType.HIGH_VOLATILITY:
                    prob = 0.15
                    if indicators.get("atr_percentile", 0.5) > 0.7:
                        prob += 0.2
                else:
                    prob = 0.1

                probabilities[regime.value] = min(prob, 1.0)

            return probabilities

        except Exception as e:
            self.logger.exception(failed("Failed to calculate transition probabilities: {e}"))
            return {}

    async def _handle_regime_transition(
        self,
        previous_detection: RegimeDetection,
        current_detection: RegimeDetection,
    ) -> None:
        """Handle regime transition."""
        try:
            transition_id = f"transition_{int(time.time())}"

            # Calculate transition duration
            duration = (
                current_detection.timestamp - previous_detection.timestamp
            ).total_seconds() / 60.0

            # Create transition record
            transition = RegimeTransition(
                transition_id=transition_id,
                timestamp=current_detection.timestamp,
                symbol=current_detection.symbol,
                timeframe=current_detection.timeframe,
                from_regime=previous_detection.current_regime,
                to_regime=current_detection.current_regime,
                transition_duration_minutes=duration,
                primary_trigger="price",  # Simplified
                trigger_strength=current_detection.confidence,
                price_at_transition=current_detection.current_price,
                volume_spike=current_detection.volume_ratio or 1.0,
                volatility_spike=current_detection.volatility_score,
            )

            # Store transition
            self.regime_transitions[transition_id] = transition

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_regime_transition(transition)

            # Update statistics
            self.tracking_stats["regime_transitions"] += 1

            self.logger.info(
                f"Regime transition: {previous_detection.current_regime.value} → {current_detection.current_regime.value}",
            )

        except Exception as e:
            self.logger.exception(failed("Failed to handle regime transition: {e}"))

    async def _store_regime_detection(self, detection: RegimeDetection) -> None:
        """Store regime detection in database."""
        try:
            data = {
                "regime_id": detection.regime_id,
                "timestamp": detection.timestamp,
                "symbol": detection.symbol,
                "timeframe": detection.timeframe,
                "current_regime": detection.current_regime.value,
                "confidence": detection.confidence,
                "duration_minutes": detection.duration_minutes,
                "price_action_score": detection.price_action_score,
                "volume_score": detection.volume_score,
                "volatility_score": detection.volatility_score,
                "current_price": detection.current_price,
                "regime_details": json.dumps(asdict(detection), default=str),
            }

            await self.storage_manager.insert_data("regime_detections", data)

        except Exception as e:
            self.logger.exception(failed("Failed to store regime detection: {e}"))
            raise

    async def _store_regime_transition(self, transition: RegimeTransition) -> None:
        """Store regime transition in database."""
        try:
            data = {
                "transition_id": transition.transition_id,
                "timestamp": transition.timestamp,
                "symbol": transition.symbol,
                "timeframe": transition.timeframe,
                "from_regime": transition.from_regime.value,
                "to_regime": transition.to_regime.value,
                "transition_duration_minutes": transition.transition_duration_minutes,
                "primary_trigger": transition.primary_trigger,
                "trigger_strength": transition.trigger_strength,
                "was_predicted": transition.was_predicted,
                "transition_details": json.dumps(asdict(transition), default=str),
            }

            await self.storage_manager.insert_data("regime_transitions", data)

        except Exception as e:
            self.logger.exception(failed("Failed to store regime transition: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="S/R level identification",
    )
    async def identify_sr_levels(
        self,
        symbol: str,
        timeframe: str,
        market_data: pd.DataFrame = None,
    ) -> list[SupportResistanceLevel]:
        """
        Identify support and resistance levels.

        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            market_data: Optional market data

        Returns:
            list[SupportResistanceLevel]: Identified S/R levels
        """
        try:
            if market_data is None:
                market_data = await self._get_market_data(symbol, timeframe)

            if market_data is None or market_data.empty:
                return []

            sr_levels = []
            current_price = float(market_data["close"].iloc[-1])

            # Identify pivot points
            pivot_levels = await self._find_pivot_levels(market_data)
            for price, strength in pivot_levels:
                level = await self._create_sr_level(
                    symbol,
                    timeframe,
                    SRLevelType.PIVOT,
                    price,
                    strength,
                    current_price,
                )
                sr_levels.append(level)

            # Identify volume-based levels
            if "volume" in market_data.columns:
                volume_levels = await self._find_volume_levels(market_data)
                for price, strength in volume_levels:
                    level = await self._create_sr_level(
                        symbol,
                        timeframe,
                        SRLevelType.VOLUME_PROFILE,
                        price,
                        strength,
                        current_price,
                    )
                    sr_levels.append(level)

            # Identify psychological levels
            psychological_levels = await self._find_psychological_levels(current_price)
            for price, strength in psychological_levels:
                level = await self._create_sr_level(
                    symbol,
                    timeframe,
                    SRLevelType.PSYCHOLOGICAL,
                    price,
                    strength,
                    current_price,
                )
                sr_levels.append(level)

            # Update active S/R levels
            key = f"{symbol}_{timeframe}"
            self.active_sr_levels[key] = sr_levels

            # Store in database
            for level in sr_levels:
                if hasattr(self, "storage_manager"):
                    await self._store_sr_level(level)

            # Update statistics
            self.tracking_stats["sr_levels_identified"] += len(sr_levels)
            self.tracking_stats["last_update"] = datetime.now()

            self.logger.debug(
                f"Identified {len(sr_levels)} S/R levels for {symbol} {timeframe}",
            )

            return sr_levels

        except Exception as e:
            self.logger.exception(
                f"Failed to identify S/R levels for {symbol} {timeframe}: {e}",
            )
            return []

    async def _find_pivot_levels(self, data: pd.DataFrame) -> list[tuple[float, float]]:
        """Find pivot-based support/resistance levels."""
        try:
            levels = []

            if len(data) < 10:
                return levels

            # Find local highs and lows
            highs = data["high"].rolling(window=5, center=True).max() == data["high"]
            lows = data["low"].rolling(window=5, center=True).min() == data["low"]

            # Get pivot high prices
            pivot_highs = data[highs]["high"].tolist()
            pivot_lows = data[lows]["low"].tolist()

            # Cluster similar levels
            all_levels = pivot_highs + pivot_lows
            clustered_levels = await self._cluster_price_levels(all_levels)

            for price, count in clustered_levels:
                strength = min(count / 5.0, 1.0)  # Normalize strength
                levels.append((price, strength))

            return levels

        except Exception as e:
            self.logger.exception(failed("Failed to find pivot levels: {e}"))
            return []

    async def _find_volume_levels(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[float, float]]:
        """Find volume-based support/resistance levels."""
        try:
            levels = []

            if len(data) < 20 or "volume" not in data.columns:
                return levels

            # Calculate VWAP levels
            (data["close"] * data["volume"]).rolling(window=20).sum() / data[
                "volume"
            ].rolling(window=20).sum()

            # Find high volume areas
            volume_threshold = data["volume"].quantile(0.8)
            high_volume_data = data[data["volume"] >= volume_threshold]

            if not high_volume_data.empty:
                high_volume_prices = high_volume_data["close"].tolist()
                clustered_levels = await self._cluster_price_levels(high_volume_prices)

                for price, count in clustered_levels:
                    strength = min(count / 3.0, 1.0)
                    levels.append((price, strength))

            return levels

        except Exception as e:
            self.logger.exception(failed("Failed to find volume levels: {e}"))
            return []

    async def _find_psychological_levels(
        self,
        current_price: float,
    ) -> list[tuple[float, float]]:
        """Find psychological support/resistance levels."""
        try:
            levels = []

            # Round numbers (e.g., 100, 1000, 10000)
            price_str = str(int(current_price))

            if len(price_str) >= 2:
                # Major round numbers
                for i in range(1, len(price_str)):
                    base = 10**i
                    lower_round = int(current_price / base) * base
                    upper_round = lower_round + base

                    # Distance-based strength
                    lower_distance = abs(current_price - lower_round) / current_price
                    upper_distance = abs(current_price - upper_round) / current_price

                    if lower_distance < 0.1:  # Within 10%
                        strength = 1.0 - lower_distance * 5
                        levels.append((float(lower_round), strength))

                    if upper_distance < 0.1:
                        strength = 1.0 - upper_distance * 5
                        levels.append((float(upper_round), strength))

            return levels

        except Exception as e:
            self.logger.exception(failed("Failed to find psychological levels: {e}"))
            return []

    async def _cluster_price_levels(
        self,
        prices: list[float],
    ) -> list[tuple[float, int]]:
        """Cluster similar price levels."""
        try:
            if not prices:
                return []

            # Sort prices
            sorted_prices = sorted(prices)
            clusters = []
            current_cluster = [sorted_prices[0]]

            threshold = (
                sorted_prices[0] * self.sr_detection_params["clustering_threshold"]
            )

            for price in sorted_prices[1:]:
                if price - current_cluster[-1] <= threshold:
                    current_cluster.append(price)
                else:
                    # Finalize current cluster
                    if len(current_cluster) >= self.sr_detection_params["min_touches"]:
                        cluster_price = sum(current_cluster) / len(current_cluster)
                        clusters.append((cluster_price, len(current_cluster)))

                    # Start new cluster
                    current_cluster = [price]
                    threshold = price * self.sr_detection_params["clustering_threshold"]

            # Handle last cluster
            if len(current_cluster) >= self.sr_detection_params["min_touches"]:
                cluster_price = sum(current_cluster) / len(current_cluster)
                clusters.append((cluster_price, len(current_cluster)))

            return clusters

        except Exception as e:
            self.logger.exception(failed("Failed to cluster price levels: {e}"))
            return []

    async def _create_sr_level(
        self,
        symbol: str,
        timeframe: str,
        level_type: SRLevelType,
        price: float,
        strength: float,
        current_price: float,
    ) -> SupportResistanceLevel:
        """Create S/R level record."""
        try:
            level_id = f"{symbol}_{timeframe}_{level_type.value}_{int(price)}_{int(time.time())}"

            # Calculate distance metrics
            distance_pct = (price - current_price) / current_price
            distance_abs = abs(price - current_price)

            # Determine if support or resistance
            if price < current_price:
                if level_type == SRLevelType.PIVOT:
                    level_type = SRLevelType.SUPPORT
            elif level_type == SRLevelType.PIVOT:
                level_type = SRLevelType.RESISTANCE

            return SupportResistanceLevel(
                level_id=level_id,
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                level_type=level_type,
                price=price,
                strength=strength,
                confidence=min(strength, 1.0),
                distance_from_current=distance_pct,
                distance_from_current_abs=distance_abs,
                is_active=True,
                breakout_probability=0.2,  # Default
                hold_probability=0.6,
                bounce_probability=0.2,
            )

        except Exception as e:
            self.logger.exception(failed("Failed to create S/R level: {e}"))
            # Return a basic level
            return SupportResistanceLevel(
                level_id="error",
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                level_type=level_type,
                price=price,
                strength=0.0,
                confidence=0.0,
            )

    async def _store_sr_level(self, level: SupportResistanceLevel) -> None:
        """Store S/R level in database."""
        try:
            data = {
                "level_id": level.level_id,
                "timestamp": level.timestamp,
                "symbol": level.symbol,
                "timeframe": level.timeframe,
                "level_type": level.level_type.value,
                "price": level.price,
                "strength": level.strength,
                "confidence": level.confidence,
                "touch_count": level.touch_count,
                "is_active": level.is_active,
                "is_broken": level.is_broken,
                "success_rate": level.success_rate,
                "level_details": json.dumps(asdict(level), default=str),
            }

            await self.storage_manager.insert_data("sr_levels", data)

        except Exception as e:
            self.logger.exception(failed("Failed to store S/R level: {e}"))
            raise

    async def _periodic_regime_detection(self) -> None:
        """Periodic regime detection task."""
        try:
            while True:
                await asyncio.sleep(self.regime_detection_interval)

                # Detect regimes for all active symbols/timeframes
                # This would typically iterate through your active trading pairs
                symbols = ["ETHUSDT", "BTCUSDT"]  # Placeholder
                timeframes = ["1h", "4h"]

                for symbol in symbols:
                    for timeframe in timeframes:
                        await self.detect_current_regime(symbol, timeframe)

        except asyncio.CancelledError:
            self.logger.info("Periodic regime detection task cancelled")
        except Exception as e:
            self.logger.exception(error("Error in periodic regime detection: {e}"))

    async def _periodic_sr_update(self) -> None:
        """Periodic S/R level update task."""
        try:
            while True:
                await asyncio.sleep(self.sr_update_interval)

                # Update S/R levels for all active symbols/timeframes
                symbols = ["ETHUSDT", "BTCUSDT"]  # Placeholder
                timeframes = ["1h", "4h"]

                for symbol in symbols:
                    for timeframe in timeframes:
                        await self.identify_sr_levels(symbol, timeframe)

        except asyncio.CancelledError:
            self.logger.info("Periodic S/R update task cancelled")
        except Exception as e:
            self.logger.exception(error("Error in periodic S/R update: {e}"))

    async def _periodic_performance_analysis(self) -> None:
        """Periodic performance analysis task."""
        try:
            while True:
                await asyncio.sleep(3600 * 6)  # Every 6 hours

                # Analyze trading performance by regime
                for regime_type in RegimeType:
                    await self.analyze_regime_performance(regime_type)

        except asyncio.CancelledError:
            self.logger.info("Periodic performance analysis task cancelled")
        except Exception as e:
            self.logger.exception(error("Error in periodic performance analysis: {e}"))

    async def analyze_regime_performance(
        self,
        regime_type: RegimeType,
        analysis_period_days: int = 7,
    ) -> TradingPerformanceByRegime | None:
        """
        Analyze trading performance for a specific regime.

        Args:
            regime_type: Regime to analyze
            analysis_period_days: Analysis period

        Returns:
            TradingPerformanceByRegime: Performance analysis or None
        """
        try:
            # This would integrate with your trade tracking system
            # For now, return placeholder data

            analysis_id = f"regime_analysis_{regime_type.value}_{int(time.time())}"

            analysis = TradingPerformanceByRegime(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                regime_type=regime_type,
                analysis_period_days=analysis_period_days,
                total_trades=0,  # Placeholder
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                average_pnl_per_trade=0.0,
                max_profit=0.0,
                max_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                average_trade_duration_minutes=0.0,
                prediction_accuracy=0.0,
                avg_confidence=0.0,
                avg_volatility=0.0,
                avg_volume=0.0,
                price_trend_strength=0.0,
            )

            # Store analysis
            self.performance_by_regime[analysis_id] = analysis

            # Update statistics
            self.tracking_stats["performance_analyses"] += 1

            return analysis

        except Exception as e:
            self.logger.exception(
                f"Failed to analyze regime performance for {regime_type.value}: {e}",
            )
            return None

    async def get_current_regime(
        self,
        symbol: str,
        timeframe: str,
    ) -> RegimeDetection | None:
        """Get current regime for symbol/timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.current_regime.get(key)

    async def get_active_sr_levels(
        self,
        symbol: str,
        timeframe: str,
    ) -> list[SupportResistanceLevel]:
        """Get active S/R levels for symbol/timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.active_sr_levels.get(key, [])

    async def get_tracking_statistics(self) -> dict[str, Any]:
        """Get comprehensive tracking statistics."""
        try:
            stats = self.tracking_stats.copy()

            # Add current state information
            stats.update(
                {
                    "active_regimes": len(self.current_regime),
                    "total_regime_detections": len(self.regime_detections),
                    "total_sr_levels": len(self.sr_levels),
                    "regime_transitions": len(self.regime_transitions),
                    "performance_analyses": len(self.performance_by_regime),
                    "cache_size": len(self.price_history) + len(self.volume_history),
                    "is_initialized": self.is_initialized,
                },
            )

            # Regime distribution
            if self.current_regime:
                regime_dist = {}
                for detection in self.current_regime.values():
                    regime = detection.current_regime.value
                    regime_dist[regime] = regime_dist.get(regime, 0) + 1
                stats["current_regime_distribution"] = regime_dist

            return stats

        except Exception as e:
            self.logger.exception(failed("Failed to get tracking statistics: {e}"))
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Regime and S/R Tracker...")

            # Clear caches
            self.price_history.clear()
            self.volume_history.clear()

            # Close storage connections
            if hasattr(self, "storage_manager"):
                await self.storage_manager.close()

            self.logger.info("Regime and S/R Tracker cleanup completed")

        except Exception as e:
            self.logger.exception(failed("Failed to cleanup Regime and S/R Tracker: {e}"))

    def combine_expert_predictions(
        self,
        expert_predictions: dict[str, float],
        strengths: dict[str, float] | None = None,
    ) -> float:
        """Combine expert predictions via simple or strength-weighted averaging.

        Args:
            expert_predictions: mapping expert_name->prediction value
            strengths: optional mapping expert_name->strength in [0,1]

        Returns:
            Combined prediction value.
        """
        try:
            weights = None
            if strengths and len(strengths) == len(expert_predictions):
                weights = {k: max(0.0, float(strengths.get(k, 0.0))) for k in expert_predictions}
            # Blend in expert predictive power if available in dispatcher
            try:
                power = {}
                if self.dispatcher_manifest and "expert_power" in self.dispatcher_manifest:
                    power = {k: float(self.dispatcher_manifest["expert_power"].get(k, 0.0)) for k in expert_predictions}
                if power:
                    # Normalize both to [0,1] and multiply
                    def _norm(d: dict[str, float]) -> dict[str, float]:
                        vals = list(d.values())
                        mx = max(vals) if vals else 0.0
                        return {k: (v / mx if mx > 0 else 0.0) for k, v in d.items()}
                    p_norm = _norm(power)
                    if weights is None:
                        weights = p_norm
                    else:
                        w_norm = _norm(weights)
                        weights = {k: w_norm.get(k, 0.0) * p_norm.get(k, 0.0) for k in expert_predictions}
            except Exception:
                pass
            if weights:
                total = sum(weights.values())
                if total <= 0:
                    # fallback to simple average
                    return float(np.mean(list(expert_predictions.values())))
                return float(sum(expert_predictions[k] * weights[k] for k in expert_predictions) / total)
            # Simple average
            return float(np.mean(list(expert_predictions.values())))
        except Exception:
            return float(np.mean(list(expert_predictions.values()))) if expert_predictions else 0.0


# Setup function for integration
async def setup_regime_sr_tracker(config: dict[str, Any]) -> RegimeSRTracker | None:
    """
    Setup and return a configured Regime and S/R Tracker instance.

    Args:
        config: Configuration dictionary

    Returns:
        RegimeSRTracker: Configured tracker instance or None if setup failed
    """
    try:
        tracker = RegimeSRTracker(config)
        if await tracker.initialize():
            return tracker
        return None
    except Exception:
        system_logger.exception(failed("Failed to setup Regime SR Tracker: {e}"))
        return None
