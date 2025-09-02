# src/components/modular_tactician.py

"""
Enhanced modular tactician with comprehensive error handling and type safety.
Provides tactical analysis and decision-making capabilities for trading strategies.
"""

import asyncio
import json
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, failed, initialization_error, invalid, missing


@dataclass
class TacticalMetrics:
    """Data class for storing tactical metrics."""
    timestamp: datetime
    entry_score: float
    exit_score: float
    position_score: float
    risk_score: float
    overall_score: float


@dataclass
class MarketCondition:
    """Data class for market conditions."""
    trend: str  # "bullish", "bearish", "sideways"
    volatility: float
    volume: float
    momentum: float
    strength: float


@dataclass
class EntrySignal:
    """Data class for entry signals."""
    symbol: str
    direction: str  # "long", "short"
    confidence: float
    price: float
    timestamp: datetime
    reasoning: str


@dataclass
class ExitSignal:
    """Data class for exit signals."""
    symbol: str
    direction: str  # "exit_long", "exit_short"
    confidence: float
    price: float
    timestamp: datetime
    reasoning: str


class ModularTactician:
    """
    Enhanced modular tactician with comprehensive error handling and type safety.
    Provides tactical analysis and decision-making for trading strategies.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the modular tactician."""
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("ModularTactician")

        # Tactician state
        self.is_tactician_active: bool = False
        self.tactician_results: Dict[str, Any] = {}
        self.tactician_history: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None

        # Configuration
        self.tactician_config: Dict[str, Any] = self.config.get("modular_tactician", {})
        self.tactician_interval: int = self.tactician_config.get("tactician_interval", 5)
        self.max_tactician_history: int = self.tactician_config.get("max_tactician_history", 100)
        self.enable_entry_monitoring: bool = self.tactician_config.get("enable_entry_monitoring", True)
        self.enable_exit_monitoring: bool = self.tactician_config.get("enable_exit_monitoring", True)
        self.enable_position_monitoring: bool = self.tactician_config.get("enable_position_monitoring", False)
        self.enable_risk_monitoring: bool = self.tactician_config.get("enable_risk_monitoring", True)

        # Market data storage
        self.market_data: Dict[str, List[Dict[str, Any]]] = {}
        self.technical_indicators: Dict[str, Dict[str, Any]] = {}
        
        # Signal tracking
        self.entry_signals: List[EntrySignal] = []
        self.exit_signals: List[ExitSignal] = []
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.entry_history: List[float] = []
        self.exit_history: List[float] = []
        self.position_history: List[float] = []
        self.risk_history: List[float] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular tactician configuration"),
            AttributeError: (False, "Missing required tactician parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular tactician initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the modular tactician."""
        try:
            self.logger.info("Initializing Modular Tactician...")

            # Load tactician configuration
            await self._load_tactician_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for modular tactician"))
                return False

            # Initialize tactician modules
            await self._initialize_tactician_modules()

            self.logger.info("✅ Modular Tactician initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Modular Tactician initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician configuration loading",
    )
    async def _load_tactician_configuration(self) -> None:
        """Load and validate tactician configuration."""
        try:
            # Set default tactician parameters
            self.tactician_config.setdefault("tactician_interval", 5)
            self.tactician_config.setdefault("max_tactician_history", 100)
            self.tactician_config.setdefault("enable_entry_monitoring", True)
            self.tactician_config.setdefault("enable_exit_monitoring", True)
            self.tactician_config.setdefault("enable_position_monitoring", False)
            self.tactician_config.setdefault("enable_risk_monitoring", True)

            # Update configuration
            self.tactician_interval = self.tactician_config["tactician_interval"]
            self.max_tactician_history = self.tactician_config["max_tactician_history"]
            self.enable_entry_monitoring = self.tactician_config["enable_entry_monitoring"]
            self.enable_exit_monitoring = self.tactician_config["enable_exit_monitoring"]
            self.enable_position_monitoring = self.tactician_config["enable_position_monitoring"]
            self.enable_risk_monitoring = self.tactician_config["enable_risk_monitoring"]

            self.logger.info("Tactician configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading tactician configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Validate tactician configuration."""
        try:
            required_keys = ["tactician_interval", "max_tactician_history"]
            for key in required_keys:
                if key not in self.tactician_config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.tactician_interval <= 0:
                self.logger.error("Tactician interval must be positive")
                return False

            if self.max_tactician_history <= 0:
                self.logger.error("Max tactician history must be positive")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician modules initialization",
    )
    async def _initialize_tactician_modules(self) -> None:
        """Initialize all tactician modules."""
        try:
            if self.enable_entry_monitoring:
                await self._initialize_entry_monitoring()

            if self.enable_exit_monitoring:
                await self._initialize_exit_monitoring()

            if self.enable_position_monitoring:
                await self._initialize_position_monitoring()

            if self.enable_risk_monitoring:
                await self._initialize_risk_monitoring()

            self.logger.info("All tactician modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing tactician modules: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="entry monitoring initialization",
    )
    async def _initialize_entry_monitoring(self) -> None:
        """Initialize entry monitoring module."""
        try:
            self.logger.info("Initializing entry monitoring module")
            # Initialize entry tracking structures
            self.entry_history = []
            self.entry_signals = []
            self.logger.info("Entry monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing entry monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exit monitoring initialization",
    )
    async def _initialize_exit_monitoring(self) -> None:
        """Initialize exit monitoring module."""
        try:
            self.logger.info("Initializing exit monitoring module")
            # Initialize exit tracking structures
            self.exit_history = []
            self.exit_signals = []
            self.logger.info("Exit monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing exit monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring initialization",
    )
    async def _initialize_position_monitoring(self) -> None:
        """Initialize position monitoring module."""
        try:
            self.logger.info("Initializing position monitoring module")
            # Initialize position tracking structures
            self.position_history = []
            self.active_positions = {}
            self.logger.info("Position monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing position monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk monitoring initialization",
    )
    async def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring module."""
        try:
            self.logger.info("Initializing risk monitoring module")
            # Initialize risk tracking structures
            self.risk_history = []
            self.logger.info("Risk monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing risk monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="tactician execution",
    )
    async def execute_tactician(self) -> bool:
        """Execute the main tactician cycle."""
        try:
            if not self.is_tactician_active:
                self.logger.warning("Tactician is not active")
                return False

            self.logger.info("Starting tactician cycle...")

            # Validate tactician inputs
            if not self._validate_tactician_inputs():
                self.logger.error("Invalid tactician inputs")
                return False

            # Perform monitoring tasks
            if self.enable_entry_monitoring:
                await self._perform_entry_monitoring()

            if self.enable_exit_monitoring:
                await self._perform_exit_monitoring()

            if self.enable_position_monitoring:
                await self._perform_position_monitoring()

            if self.enable_risk_monitoring:
                await self._perform_risk_monitoring()

            # Store tactician results
            await self._store_tactician_results()

            self.logger.info("Tactician cycle completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing tactician: {e}")
            return False

    def _validate_tactician_inputs(self) -> bool:
        """Validate inputs for tactician execution."""
        try:
            if not hasattr(self, 'tactician_config'):
                self.logger.error("Tactician configuration not available")
                return False

            if not hasattr(self, 'market_data'):
                self.logger.error("Market data not available")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating tactician inputs: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="entry monitoring",
    )
    async def _perform_entry_monitoring(self) -> None:
        """Perform entry monitoring tasks."""
        try:
            self.logger.debug("Performing entry monitoring...")

            # Analyze market conditions for entry opportunities
            for symbol in self.market_data.keys():
                if symbol not in self.active_positions:
                    entry_signal = await self._analyze_entry_opportunity(symbol)
                    if entry_signal and entry_signal.confidence > 0.7:
                        self.entry_signals.append(entry_signal)
                        self.logger.info(f"Entry signal generated for {symbol}: {entry_signal.direction}")

            # Calculate entry score
            entry_score = self._calculate_entry_score()
            self.entry_history.append(entry_score)
            if len(self.entry_history) > self.max_tactician_history:
                self.entry_history.pop(0)

            self.logger.debug(f"Entry monitoring completed. Score: {entry_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing entry monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exit monitoring",
    )
    async def _perform_exit_monitoring(self) -> None:
        """Perform exit monitoring tasks."""
        try:
            self.logger.debug("Performing exit monitoring...")

            # Analyze existing positions for exit opportunities
            for symbol, position in self.active_positions.items():
                exit_signal = await self._analyze_exit_opportunity(symbol, position)
                if exit_signal and exit_signal.confidence > 0.7:
                    self.exit_signals.append(exit_signal)
                    self.logger.info(f"Exit signal generated for {symbol}: {exit_signal.direction}")

            # Calculate exit score
            exit_score = self._calculate_exit_score()
            self.exit_history.append(exit_score)
            if len(self.exit_history) > self.max_tactician_history:
                self.exit_history.pop(0)

            self.logger.debug(f"Exit monitoring completed. Score: {exit_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing exit monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring",
    )
    async def _perform_position_monitoring(self) -> None:
        """Perform position monitoring tasks."""
        try:
            self.logger.debug("Performing position monitoring...")

            # Monitor active positions
            for symbol, position in self.active_positions.items():
                position_health = await self._monitor_position_health(symbol, position)
                if position_health < 0.5:  # Low health threshold
                    self.logger.warning(f"Position health low for {symbol}: {position_health:.2f}")

            # Calculate position score
            position_score = self._calculate_position_score()
            self.position_history.append(position_score)
            if len(self.position_history) > self.max_tactician_history:
                self.position_history.pop(0)

            self.logger.debug(f"Position monitoring completed. Score: {position_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing position monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk monitoring",
    )
    async def _perform_risk_monitoring(self) -> None:
        """Perform risk monitoring tasks."""
        try:
            self.logger.debug("Performing risk monitoring...")

            # Calculate risk metrics
            portfolio_risk = self._calculate_portfolio_risk()
            position_risk = self._calculate_position_risk()
            market_risk = self._calculate_market_risk()
            correlation_risk = self._calculate_correlation_risk()

            # Store risk metrics
            risk_score = self._calculate_risk_score(
                portfolio_risk, position_risk, market_risk, correlation_risk
            )

            self.risk_history.append(risk_score)
            if len(self.risk_history) > self.max_tactician_history:
                self.risk_history.pop(0)

            self.logger.debug(f"Risk monitoring completed. Score: {risk_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing risk monitoring: {e}")

    # Entry monitoring analysis methods

    async def _analyze_entry_opportunity(self, symbol: str) -> Optional[EntrySignal]:
        """Analyze entry opportunity for a symbol."""
        try:
            if symbol not in self.market_data or not self.market_data[symbol]:
                return None

            # Get latest market data
            latest_data = self.market_data[symbol][-1]
            
            # Analyze price action
            price_action_score = self._analyze_price_action(symbol)
            
            # Analyze volume
            volume_score = self._analyze_volume(symbol)
            
            # Analyze momentum indicators
            momentum_score = self._analyze_momentum_indicators(symbol)
            
            # Analyze support/resistance
            support_resistance_score = self._analyze_support_resistance(symbol)
            
            # Calculate overall entry score
            entry_score = (price_action_score + volume_score + momentum_score + support_resistance_score) / 4
            
            # Determine direction based on analysis
            direction = "long" if entry_score > 0.6 else "short" if entry_score < 0.4 else "neutral"
            
            if direction != "neutral" and entry_score > 0.7:
                return EntrySignal(
                    symbol=symbol,
                    direction=direction,
                    confidence=entry_score,
                    price=latest_data.get("close", 0.0),
                    timestamp=datetime.now(),
                    reasoning=f"Entry signal based on technical analysis (score: {entry_score:.2f})"
                )
            
            return None

        except Exception as e:
            self.logger.error(error(f"Error analyzing entry opportunity for {symbol}: {e}"))
            return None

    def _analyze_price_action(self, symbol: str) -> float:
        """Analyze price action patterns."""
        try:
            # Placeholder implementation - replace with actual price action analysis
            # This should include pattern recognition, candlestick analysis, etc.
            return 0.5  # Neutral score
        except Exception as e:
            self.logger.error(error(f"Error analyzing price action: {e}"))
            return 0.0

    def _analyze_volume(self, symbol: str) -> float:
        """Analyze volume patterns."""
        try:
            # Placeholder implementation - replace with actual volume analysis
            # This should include volume trends, volume-price relationships, etc.
            return 0.5  # Neutral score
        except Exception as e:
            self.logger.error(error(f"Error analyzing volume: {e}"))
            return 0.0

    def _analyze_momentum_indicators(self, symbol: str) -> float:
        """Analyze momentum indicators."""
        try:
            # Placeholder implementation - replace with actual momentum analysis
            # This should include RSI, MACD, stochastic, etc.
            return 0.5  # Neutral score
        except Exception as e:
            self.logger.error(error(f"Error analyzing momentum indicators: {e}"))
            return 0.0

    def _analyze_support_resistance(self, symbol: str) -> float:
        """Analyze support and resistance levels."""
        try:
            # Placeholder implementation - replace with actual support/resistance analysis
            # This should include pivot points, Fibonacci levels, etc.
            return 0.5  # Neutral score
        except Exception as e:
            self.logger.error(error(f"Error analyzing support/resistance: {e}"))
            return 0.0

    # Exit monitoring tracking methods

    async def _analyze_exit_opportunity(self, symbol: str, position: Dict[str, Any]) -> Optional[ExitSignal]:
        """Analyze exit opportunity for a position."""
        try:
            if symbol not in self.market_data or not self.market_data[symbol]:
                return None

            # Get latest market data
            latest_data = self.market_data[symbol][-1]
            position_type = position.get("type", "long")
            
            # Track stop loss
            stop_loss_triggered = self._track_stop_loss(symbol, position)
            
            # Track take profit
            take_profit_triggered = self._track_take_profit(symbol, position)
            
            # Track trailing stop
            trailing_stop_triggered = self._track_trailing_stop(symbol, position)
            
            # Track time-based exit
            time_exit_triggered = self._track_time_based_exit(symbol, position)
            
            # Determine exit direction
            exit_direction = f"exit_{position_type}"
            
            # Calculate exit confidence
            exit_confidence = 0.0
            if stop_loss_triggered:
                exit_confidence = 0.9
            elif take_profit_triggered:
                exit_confidence = 0.8
            elif trailing_stop_triggered:
                exit_confidence = 0.7
            elif time_exit_triggered:
                exit_confidence = 0.6
            
            if exit_confidence > 0.6:
                return ExitSignal(
                    symbol=symbol,
                    direction=exit_direction,
                    confidence=exit_confidence,
                    price=latest_data.get("close", 0.0),
                    timestamp=datetime.now(),
                    reasoning=f"Exit signal based on {exit_direction} (confidence: {exit_confidence:.2f})"
                )
            
            return None

        except Exception as e:
            self.logger.error(error(f"Error analyzing exit opportunity for {symbol}: {e}"))
            return None

    def _track_stop_loss(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Track stop loss conditions."""
        try:
            # Placeholder implementation - replace with actual stop loss tracking
            # This should check if current price has hit the stop loss level
            return False
        except Exception as e:
            self.logger.error(error(f"Error tracking stop loss: {e}"))
            return False

    def _track_take_profit(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Track take profit conditions."""
        try:
            # Placeholder implementation - replace with actual take profit tracking
            # This should check if current price has hit the take profit level
            return False
        except Exception as e:
            self.logger.error(error(f"Error tracking take profit: {e}"))
            return False

    def _track_trailing_stop(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Track trailing stop conditions."""
        try:
            # Placeholder implementation - replace with actual trailing stop tracking
            # This should dynamically adjust stop loss based on price movement
            return False
        except Exception as e:
            self.logger.error(error(f"Error tracking trailing stop: {e}"))
            return False

    def _track_time_based_exit(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Track time-based exit conditions."""
        try:
            # Placeholder implementation - replace with actual time-based exit tracking
            # This should check if position has been held for too long
            return False
        except Exception as e:
            self.logger.error(error(f"Error tracking time-based exit: {e}"))
            return False

    # Position monitoring methods

    async def _monitor_position_health(self, symbol: str, position: Dict[str, Any]) -> float:
        """Monitor the health of a position."""
        try:
            # Placeholder implementation - replace with actual position health monitoring
            # This should consider factors like drawdown, time in position, etc.
            return 0.8  # Good health score
        except Exception as e:
            self.logger.error(error(f"Error monitoring position health: {e}"))
            return 0.0

    def _track_position_size(self, symbol: str, position: Dict[str, Any]) -> float:
        """Track position size and exposure."""
        try:
            # Placeholder implementation - replace with actual position size tracking
            # This should monitor position sizing relative to portfolio
            return 0.5  # Neutral score
        except Exception as e:
            self.logger.error(error(f"Error tracking position size: {e}"))
            return 0.0

    def _monitor_exposure_limits(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Monitor exposure limits."""
        try:
            # Placeholder implementation - replace with actual exposure limit monitoring
            # This should check if position exceeds risk limits
            return True  # Within limits
        except Exception as e:
            self.logger.error(error(f"Error monitoring exposure limits: {e}"))
            return False

    def _monitor_correlation(self, symbol: str, position: Dict[str, Any]) -> float:
        """Monitor correlation with other positions."""
        try:
            # Placeholder implementation - replace with actual correlation monitoring
            # This should check correlation with existing positions
            return 0.3  # Low correlation
        except Exception as e:
            self.logger.error(error(f"Error monitoring correlation: {e}"))
            return 0.0

    def _monitor_concentration_limits(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Monitor concentration limits."""
        try:
            # Placeholder implementation - replace with actual concentration limit monitoring
            # This should check if position is too concentrated in one asset
            return True  # Within limits
        except Exception as e:
            self.logger.error(error(f"Error monitoring concentration limits: {e}"))
            return False

    # Risk monitoring methods

    def _calculate_portfolio_risk(self) -> float:
        """Calculate portfolio risk metrics."""
        try:
            # Placeholder implementation - replace with actual portfolio risk calculation
            # This should include VaR, CVaR, volatility, etc.
            return 0.2  # 20% risk level
        except Exception as e:
            self.logger.error(error(f"Error calculating portfolio risk: {e}"))
            return 0.0

    def _calculate_position_risk(self) -> float:
        """Calculate individual position risk."""
        try:
            # Placeholder implementation - replace with actual position risk calculation
            # This should include position-specific risk metrics
            return 0.15  # 15% risk level
        except Exception as e:
            self.logger.error(error(f"Error calculating position risk: {e}"))
            return 0.0

    def _calculate_market_risk(self) -> float:
        """Calculate market risk metrics."""
        try:
            # Placeholder implementation - replace with actual market risk calculation
            # This should include market volatility, sector risk, etc.
            return 0.25  # 25% risk level
        except Exception as e:
            self.logger.error(error(f"Error calculating market risk: {e}"))
            return 0.0

    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk."""
        try:
            # Placeholder implementation - replace with actual correlation risk calculation
            # This should measure portfolio correlation risk
            return 0.1  # 10% correlation risk
        except Exception as e:
            self.logger.error(error(f"Error calculating correlation risk: {e}"))
            return 0.0

    def _calculate_risk_score(self, portfolio_risk: float, position_risk: float, 
                             market_risk: float, correlation_risk: float) -> float:
        """Calculate overall risk score."""
        try:
            # Simple risk scoring algorithm - replace with more sophisticated logic
            # Lower score = higher risk
            total_risk = portfolio_risk + position_risk + market_risk + correlation_risk
            risk_score = max(0.0, 100.0 - (total_risk * 100.0))
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 100.0

    # Scoring methods

    def _calculate_entry_score(self) -> float:
        """Calculate overall entry score."""
        try:
            if not self.entry_signals:
                return 0.0
            
            # Calculate average confidence of recent entry signals
            recent_signals = self.entry_signals[-10:]  # Last 10 signals
            if recent_signals:
                avg_confidence = sum(signal.confidence for signal in recent_signals) / len(recent_signals)
                return avg_confidence * 100.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating entry score: {e}")
            return 0.0

    def _calculate_exit_score(self) -> float:
        """Calculate overall exit score."""
        try:
            if not self.exit_signals:
                return 0.0
            
            # Calculate average confidence of recent exit signals
            recent_signals = self.exit_signals[-10:]  # Last 10 signals
            if recent_signals:
                avg_confidence = sum(signal.confidence for signal in recent_signals) / len(recent_signals)
                return avg_confidence * 100.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating exit score: {e}")
            return 0.0

    def _calculate_position_score(self) -> float:
        """Calculate overall position score."""
        try:
            if not self.active_positions:
                return 100.0  # No positions = perfect score
            
            # Calculate average position health
            health_scores = []
            for symbol, position in self.active_positions.items():
                # Placeholder health calculation
                health = 0.8  # Default health
                health_scores.append(health)
            
            if health_scores:
                avg_health = sum(health_scores) / len(health_scores)
                return avg_health * 100.0
            
            return 100.0
            
        except Exception as e:
            self.logger.error(f"Error calculating position score: {e}")
            return 100.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician results storage",
    )
    async def _store_tactician_results(self) -> None:
        """Store tactician results and history."""
        try:
            timestamp = datetime.now()
            
            # Create tactician result
            result = {
                "timestamp": timestamp.isoformat(),
                "entry_score": self.entry_history[-1] if self.entry_history else 0.0,
                "exit_score": self.exit_history[-1] if self.exit_history else 0.0,
                "position_score": self.position_history[-1] if self.position_history else 0.0,
                "risk_score": self.risk_history[-1] if self.risk_history else 0.0,
                "is_tactician_active": self.is_tactician_active,
                "tactician_interval": self.tactician_interval,
                "active_positions_count": len(self.active_positions),
                "entry_signals_count": len(self.entry_signals),
                "exit_signals_count": len(self.exit_signals),
            }
            
            # Store current result
            self.tactician_results = result
            
            # Add to history
            self.tactician_history.append(result)
            
            # Maintain history size
            if len(self.tactician_history) > self.max_tactician_history:
                self.tactician_history.pop(0)
            
            self.logger.debug("Tactician results stored successfully")
            
        except Exception as e:
            self.logger.error(f"Error storing tactician results: {e}")

    def get_tactician_results(self) -> Optional[Dict[str, Any]]:
        """Get current tactician results."""
        try:
            return self.tactician_results.copy() if self.tactician_results else None
        except Exception as e:
            self.logger.error(f"Error getting tactician results: {e}")
            return None

    def get_tactician_history(self) -> List[Dict[str, Any]]:
        """Get tactician history."""
        try:
            return self.tactician_history.copy()
        except Exception as e:
            self.logger.error(error(f"Error getting tactician history: {e}"))
            return []

    def get_tactician_status(self) -> Dict[str, Any]:
        """Get tactician status."""
        try:
            return {
                "is_tactician_active": self.is_tactician_active,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "tactician_interval": self.tactician_interval,
                "max_tactician_history": self.max_tactician_history,
                "enable_entry_monitoring": self.enable_entry_monitoring,
                "enable_exit_monitoring": self.enable_exit_monitoring,
                "enable_position_monitoring": self.enable_position_monitoring,
                "enable_risk_monitoring": self.enable_risk_monitoring,
                "entry_history_size": len(self.entry_history),
                "exit_history_size": len(self.exit_history),
                "position_history_size": len(self.position_history),
                "risk_history_size": len(self.risk_history),
                "tactician_history_size": len(self.tactician_history),
                "active_positions_count": len(self.active_positions),
                "entry_signals_count": len(self.entry_signals),
                "exit_signals_count": len(self.exit_signals),
            }
        except Exception as e:
            self.logger.error(f"Error getting tactician status: {e}")
            return {}

    async def start(self) -> bool:
        """Start the tactician."""
        try:
            if self.is_tactician_active:
                self.logger.warning("Tactician is already running")
                return True
            
            self.logger.info("🚀 Starting Modular Tactician...")
            self.is_tactician_active = True
            self.start_time = datetime.now()
            
            # Start tactician loop
            asyncio.create_task(self._tactician_loop())
            
            self.logger.info("✅ Modular Tactician started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting tactician: {e}")
            self.is_tactician_active = False
            return False

    async def _tactician_loop(self) -> None:
        """Main tactician loop."""
        try:
            while self.is_tactician_active:
                await self.execute_tactician()
                await asyncio.sleep(self.tactician_interval)
                
        except Exception as e:
            self.logger.error(f"Error in tactician loop: {e}")
            self.is_tactician_active = False

    async def stop(self) -> None:
        """Stop the tactician."""
        try:
            self.logger.info("🛑 Stopping Modular Tactician...")
            self.is_tactician_active = False
            
            # Wait for tactician loop to finish
            await asyncio.sleep(1)
            
            self.logger.info("✅ Modular Tactician stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping tactician: {e}")

    async def setup_modular_tactician(self) -> None:
        """Setup function for modular tactician."""
        try:
            # Initialize the tactician
            if not await self.initialize():
                raise RuntimeError("Failed to initialize modular tactician")
            
            # Start the tactician
            if not await self.start():
                raise RuntimeError("Failed to start modular tactician")
            
            self.logger.info("Modular tactician setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in modular tactician setup: {e}")
            raise


# Factory function for creating modular tactician instances
async def create_modular_tactician(config: Dict[str, Any]) -> ModularTactician:
    """Create and setup a modular tactician instance."""
    try:
        tactician = ModularTactician(config)
        await tactician.setup_modular_tactician()
        return tactician
    except Exception as e:
        system_logger.error(f"Failed to create modular tactician: {e}")
        raise
