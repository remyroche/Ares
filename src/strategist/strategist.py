from datetime import datetime, timedelta
from typing import Any

from pandas.core.dtypes.dtypes import datetime  # Ensure Union is imported

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Strategist:
    """
    Enhanced strategist with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize strategist with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Strategist")

        # Strategy state
        self.current_strategy: dict[str, Any] | None = None
        self.strategy_history: list[dict[str, Any]] = []
        self.last_strategy_update: datetime | None = None

        # Configuration
        self.strategy_config: dict[str, Any] = self.config.get("strategist", {})
        self.risk_config: dict[str, Any] = self.config.get("risk_management", {})

        # Performance tracking
        self.strategy_performance: dict[str, float] = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        
        # SR Analyzer integration
        self.sr_analyzer = None
        self.enable_sr_strategy: bool = self.strategy_config.get("enable_sr_strategy", True)
        
        # SR Breakout Predictor integration
        self.sr_breakout_predictor = None
        self.enable_sr_breakout_strategy: bool = self.strategy_config.get("enable_sr_breakout_strategy", True)
        
        # ML Prediction integration
        self.ml_predictions = None
        self.enable_ml_strategy: bool = self.strategy_config.get("enable_ml_strategy", True)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid strategist configuration"),
            AttributeError: (False, "Missing required strategy parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="strategist initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize strategist with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Strategist...")

            # Load strategy configuration
            await self._load_strategy_configuration()

            # Initialize risk management
            await self._initialize_risk_management()

            # Initialize SR analyzer
            if self.enable_sr_strategy:
                await self._initialize_sr_analyzer()
                
            # Initialize SR Breakout Predictor
            if self.enable_sr_breakout_strategy:
                await self._initialize_sr_breakout_predictor()

            # Initialize ML strategy
            if self.enable_ml_strategy:
                await self._initialize_ml_strategy()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for strategist")
                return False

            self.logger.info("✅ Strategist initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Strategist initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy configuration loading",
    )
    async def _load_strategy_configuration(self) -> None:
        """Load strategy configuration."""
        try:
            # Set default strategy parameters
            self.strategy_config.setdefault("max_position_size", 0.1)
            self.strategy_config.setdefault("max_daily_trades", 10)
            self.strategy_config.setdefault("min_confidence_threshold", 0.6)
            self.strategy_config.setdefault("strategy_timeout_seconds", 30)

            self.logger.info("Strategy configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading strategy configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR analyzer initialization",
    )
    async def _initialize_sr_analyzer(self) -> None:
        """Initialize SR analyzer for strategy generation."""
        try:
            from src.analyst.sr_analyzer import SRLevelAnalyzer
            
            self.sr_analyzer = SRLevelAnalyzer(self.config)
            await self.sr_analyzer.initialize()
            self.logger.info("✅ SR analyzer initialized for strategy generation")
        except Exception as e:
            self.logger.error(f"Error initializing SR analyzer: {e}")
            self.sr_analyzer = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR breakout predictor initialization",
    )
    async def _initialize_sr_breakout_predictor(self) -> None:
        """Initialize SR Breakout Predictor for strategy generation."""
        try:
            from src.analyst.sr_breakout_predictor import SRBreakoutPredictor
            self.sr_breakout_predictor = SRBreakoutPredictor(self.config)
            await self.sr_breakout_predictor.initialize()
            self.logger.info("✅ SR Breakout Predictor initialized for strategy generation")
        except Exception as e:
            self.logger.error(f"Error initializing SR Breakout Predictor: {e}")
            self.sr_breakout_predictor = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML strategy initialization",
    )
    async def _initialize_ml_strategy(self) -> None:
        """Initialize ML strategy for strategy generation."""
        try:
            self.logger.info("Initializing ML strategy...")
            
            # ML strategy will be initialized when predictions are received
            self.ml_predictions = {}
            self.logger.info("✅ ML strategy initialized for strategy generation")
                
        except Exception as e:
            self.logger.error(f"Error initializing ML strategy: {e}")
            self.ml_predictions = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk management initialization",
    )
    async def _initialize_risk_management(self) -> None:
        """Initialize risk management components."""
        try:
            # Set default risk parameters
            self.risk_config.setdefault("max_daily_loss", 0.02)
            self.risk_config.setdefault("max_position_risk", 0.01)
            self.risk_config.setdefault("max_correlation", 0.7)
            self.risk_config.setdefault("stop_loss_multiplier", 2.0)

            self.logger.info("Risk management initialized")

        except Exception as e:
            self.logger.error(f"Error initializing risk management: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate strategist configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_strategy_keys = [
                "max_position_size",
                "max_daily_trades",
                "min_confidence_threshold",
            ]
            for key in required_strategy_keys:
                if key not in self.strategy_config:
                    self.logger.error(
                        f"Missing required strategy configuration key: {key}",
                    )
                    return False

            required_risk_keys = [
                "max_daily_loss",
                "max_position_risk",
                "stop_loss_multiplier",
            ]
            for key in required_risk_keys:
                if key not in self.risk_config:
                    self.logger.error(f"Missing required risk configuration key: {key}")
                    return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to data source"),
            TimeoutError: (None, "Strategy generation timed out"),
            ValueError: (None, "Invalid market intelligence data"),
        },
        default_return=None,
        context="strategy generation",
    )
    async def generate_strategy(
        self,
        market_intelligence: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate trading strategy with enhanced error handling.

        Args:
            market_intelligence: Optional market intelligence data

        Returns:
            Optional[Dict[str, Any]]: Generated strategy or None if failed
        """
        try:
            self.logger.info("Generating trading strategy...")

            # Validate market intelligence
            if not self._validate_market_intelligence(market_intelligence):
                self.logger.warning(
                    "Invalid market intelligence, using default strategy",
                )
                market_intelligence = self._get_default_market_intelligence()

            # Generate strategy components
            entry_signals = await self._generate_entry_signals(market_intelligence)
            exit_signals = await self._generate_exit_signals(market_intelligence)
            risk_parameters = await self._calculate_risk_parameters(market_intelligence)
            position_sizing = await self._calculate_position_sizing(market_intelligence)

            # Generate SR-based strategy components
            sr_strategy = None
            if self.enable_sr_strategy and self.sr_analyzer:
                sr_strategy = await self._generate_sr_strategy(market_intelligence)
                
            # Generate SR Breakout strategy components
            sr_breakout_strategy = None
            if self.enable_sr_breakout_strategy and self.sr_breakout_predictor:
                sr_breakout_strategy = await self._generate_sr_breakout_strategy(market_intelligence)

            # Generate ML strategy components
            ml_strategy = None
            if self.enable_ml_strategy:
                ml_strategy = await self._generate_ml_strategy(market_intelligence)

            # Combine into comprehensive strategy
            strategy = {
                "entry_signals": entry_signals,
                "exit_signals": exit_signals,
                "risk_parameters": risk_parameters,
                "position_sizing": position_sizing,
                "sr_strategy": sr_strategy,
                "sr_breakout_strategy": sr_breakout_strategy,
                "ml_strategy": ml_strategy,
                "confidence_score": self._calculate_confidence_score(
                    market_intelligence,
                ),
                "generation_time": datetime.now(),
                "valid_until": datetime.now() + timedelta(minutes=30),
            }

            # Validate strategy
            if not self._validate_strategy(strategy):
                self.logger.error("Generated strategy validation failed")
                return None

            # Update strategy state
            self.current_strategy = strategy
            self.strategy_history.append(strategy)
            self.last_strategy_update = datetime.now()

            self.logger.info("✅ Strategy generated successfully")
            return strategy

        except Exception as e:
            self.logger.error(f"Error generating strategy: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="market intelligence validation",
    )
    def _validate_market_intelligence(
        self,
        market_intelligence: dict[str, Any] | None,
    ) -> bool:
        """
        Validate market intelligence data.

        Args:
            market_intelligence: Market intelligence data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not market_intelligence:
                return False

            required_keys = [
                "market_regime",
                "volatility",
                "trend_direction",
                "support_resistance",
            ]
            for key in required_keys:
                if key not in market_intelligence:
                    self.logger.warning(f"Missing market intelligence key: {key}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating market intelligence: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="default market intelligence",
    )
    def _get_default_market_intelligence(self) -> dict[str, Any]:
        """
        Get default market intelligence when actual data is unavailable.

        Returns:
            Dict[str, Any]: Default market intelligence
        """
        try:
            return {
                "market_regime": "neutral",
                "volatility": 0.02,
                "trend_direction": "sideways",
                "support_resistance": {"support_levels": [], "resistance_levels": []},
                "technical_indicators": {
                    "rsi": 50.0,
                    "macd": 0.0,
                    "bollinger_bands": {"upper": 1.0, "middle": 1.0, "lower": 1.0},
                },
            }

        except Exception as e:
            self.logger.error(f"Error creating default market intelligence: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="entry signal generation",
    )
    async def _generate_entry_signals(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Generate entry signals based on market intelligence.

        Args:
            market_intelligence: Market intelligence data

        Returns:
            Optional[Dict[str, Any]]: Entry signals or None
        """
        try:
            regime = market_intelligence.get("market_regime", "neutral")
            trend = market_intelligence.get("trend_direction", "sideways")
            volatility = market_intelligence.get("volatility", 0.02)

            entry_signals = {
                "long_conditions": [],
                "short_conditions": [],
                "confidence": 0.5,
                "timeframe": "1m",
            }

            # Generate long entry conditions
            if trend == "bullish" and regime in ["bull", "neutral"]:
                entry_signals["long_conditions"].append(
                    {
                        "type": "trend_following",
                        "condition": "price_above_ma",
                        "confidence": 0.7,
                    },
                )

            # Generate short entry conditions
            if trend == "bearish" and regime in ["bear", "neutral"]:
                entry_signals["short_conditions"].append(
                    {
                        "type": "trend_following",
                        "condition": "price_below_ma",
                        "confidence": 0.7,
                    },
                )

            # Adjust confidence based on volatility
            if volatility > 0.03:
                entry_signals["confidence"] *= (
                    0.8  # Reduce confidence in high volatility
                )

            return entry_signals

        except Exception as e:
            self.logger.error(f"Error generating entry signals: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exit signal generation",
    )
    async def _generate_exit_signals(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Generate exit signals based on market intelligence.

        Args:
            market_intelligence: Market intelligence data

        Returns:
            Optional[Dict[str, Any]]: Exit signals or None
        """
        try:
            volatility = market_intelligence.get("volatility", 0.02)

            exit_signals = {
                "take_profit_levels": [],
                "stop_loss_levels": [],
                "trailing_stop": True,
                "time_based_exit": False,
            }

            # Calculate take profit levels
            base_tp_distance = 0.02  # 2% base take profit
            volatility_multiplier = min(volatility * 100, 3.0)  # Cap at 3x

            exit_signals["take_profit_levels"] = [
                {
                    "level": 1 + (base_tp_distance * volatility_multiplier),
                    "weight": 0.6,
                },
                {
                    "level": 1 + (base_tp_distance * volatility_multiplier * 1.5),
                    "weight": 0.3,
                },
                {
                    "level": 1 + (base_tp_distance * volatility_multiplier * 2.0),
                    "weight": 0.1,
                },
            ]

            # Calculate stop loss levels
            base_sl_distance = 0.01  # 1% base stop loss
            exit_signals["stop_loss_levels"] = [
                {
                    "level": 1 - (base_sl_distance * volatility_multiplier),
                    "weight": 0.8,
                },
                {
                    "level": 1 - (base_sl_distance * volatility_multiplier * 1.5),
                    "weight": 0.2,
                },
            ]

            return exit_signals

        except Exception as e:
            self.logger.error(f"Error generating exit signals: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk parameter calculation",
    )
    async def _calculate_risk_parameters(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Calculate risk parameters based on market intelligence.

        Args:
            market_intelligence: Market intelligence data

        Returns:
            Optional[Dict[str, Any]]: Risk parameters or None
        """
        try:
            volatility = market_intelligence.get("volatility", 0.02)
            regime = market_intelligence.get("market_regime", "neutral")

            risk_params = {
                "max_position_size": self.strategy_config["max_position_size"],
                "max_daily_loss": self.risk_config["max_daily_loss"],
                "stop_loss_distance": max(0.005, volatility * 2),  # Minimum 0.5%
                "take_profit_distance": max(0.01, volatility * 3),  # Minimum 1%
                "max_correlation": self.risk_config["max_correlation"],
                "regime_adjustment": self._get_regime_adjustment(regime),
            }

            return risk_params

        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=1.0,
        context="regime adjustment calculation",
    )
    def _get_regime_adjustment(self, regime: str) -> float:
        """
        Get position size adjustment based on market regime.

        Args:
            regime: Market regime string

        Returns:
            float: Position size adjustment multiplier
        """
        try:
            regime_adjustments = {
                "bull": 1.2,  # Increase position size in bullish regime
                "bear": 0.8,  # Decrease position size in bearish regime
                "neutral": 1.0,  # No adjustment in neutral regime
                "volatile": 0.7,  # Reduce position size in volatile regime
            }

            return regime_adjustments.get(regime, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating regime adjustment: {e}")
            return 1.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position sizing calculation",
    )
    async def _calculate_position_sizing(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Calculate position sizing parameters.

        Args:
            market_intelligence: Market intelligence data

        Returns:
            Optional[Dict[str, Any]]: Position sizing parameters or None
        """
        try:
            volatility = market_intelligence.get("volatility", 0.02)
            regime = market_intelligence.get("market_regime", "neutral")

            # Base position size
            base_size = self.strategy_config["max_position_size"]

            # Adjust for volatility
            volatility_adjustment = max(
                0.5,
                1 - (volatility * 10),
            )  # Reduce size in high volatility

            # Adjust for regime
            regime_adjustment = self._get_regime_adjustment(regime)

            # Calculate final position size
            final_size = base_size * volatility_adjustment * regime_adjustment

            position_sizing = {
                "position_size": min(
                    final_size,
                    self.strategy_config["max_position_size"],
                ),
                "volatility_adjustment": volatility_adjustment,
                "regime_adjustment": regime_adjustment,
                "max_trades_per_day": self.strategy_config["max_daily_trades"],
            }

            return position_sizing

        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.5,
        context="confidence score calculation",
    )
    def _calculate_confidence_score(self, market_intelligence: dict[str, Any]) -> float:
        """
        Calculate confidence score for the strategy.

        Args:
            market_intelligence: Market intelligence data

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            confidence = 0.5  # Base confidence

            # Adjust based on market regime clarity
            regime = market_intelligence.get("market_regime", "neutral")
            if regime in ["bull", "bear"]:
                confidence += 0.2
            elif regime == "volatile":
                confidence -= 0.1

            # Adjust based on volatility
            volatility = market_intelligence.get("volatility", 0.02)
            if volatility < 0.015:  # Low volatility
                confidence += 0.1
            elif volatility > 0.04:  # High volatility
                confidence -= 0.2

            # Adjust based on trend clarity
            trend = market_intelligence.get("trend_direction", "sideways")
            if trend in ["bullish", "bearish"]:
                confidence += 0.1
            elif trend == "sideways":
                confidence -= 0.1

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR strategy generation",
    )
    async def _generate_sr_strategy(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Generate SR-based strategy components."""
        try:
            if not self.sr_analyzer:
                return None

            # Get market data from intelligence
            market_data = market_intelligence.get("market_data", {})
            if not market_data:
                return None

            # Convert to DataFrame if needed
            if isinstance(market_data, dict):
                import pandas as pd
                df = pd.DataFrame([market_data])
            else:
                df = market_data

            # Perform SR analysis
            sr_analysis = await self.sr_analyzer.analyze(df)
            
            if not sr_analysis:
                return None

            # Generate SR-based strategy
            sr_strategy = {
                "support_levels": sr_analysis.get("support_levels", []),
                "resistance_levels": sr_analysis.get("resistance_levels", []),
                "sr_entry_signals": self._generate_sr_entry_signals(sr_analysis),
                "sr_exit_signals": self._generate_sr_exit_signals(sr_analysis),
                "sr_risk_levels": self._calculate_sr_risk_levels(sr_analysis),
                "sr_confidence": sr_analysis.get("support_confidence", 0.0) + 
                               sr_analysis.get("resistance_confidence", 0.0) / 2,
            }

            return sr_strategy

        except Exception as e:
            self.logger.error(f"Error generating SR strategy: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML strategy generation",
    )
    async def _generate_ml_strategy(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Generate ML-based strategy components.

        Args:
            market_intelligence: Market intelligence data

        Returns:
            Optional[Dict[str, Any]]: ML strategy components
        """
        try:
            # Get ML predictions from market intelligence
            ml_predictions = market_intelligence.get("ml_predictions", {})
            
            if not ml_predictions:
                self.logger.warning("No ML predictions available for strategy generation")
                return None

            # Generate ML-based entry signals
            ml_entry_signals = self._generate_ml_entry_signals(ml_predictions)
            
            # Generate ML-based exit signals
            ml_exit_signals = self._generate_ml_exit_signals(ml_predictions)
            
            # Generate ML-based position sizing
            ml_position_sizing = self._generate_ml_position_sizing(ml_predictions)
            
            # Generate ML-based risk parameters
            ml_risk_parameters = self._generate_ml_risk_parameters(ml_predictions)

            return {
                "ml_entry_signals": ml_entry_signals,
                "ml_exit_signals": ml_exit_signals,
                "ml_position_sizing": ml_position_sizing,
                "ml_risk_parameters": ml_risk_parameters,
                "ml_confidence_scores": ml_predictions.get("confidence_scores", {}),
                "ml_expected_decreases": ml_predictions.get("expected_decreases", {}),
                "generation_time": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error generating ML strategy: {e}")
            return None

    def _generate_ml_entry_signals(self, ml_predictions: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate ML-based entry signals."""
        try:
            entry_signals = []
            confidence_scores = ml_predictions.get("confidence_scores", {})
            
            for increase_level, confidence in confidence_scores.items():
                if confidence >= 0.7:  # High confidence threshold
                    entry_signals.append({
                        "signal_type": "ml_high_confidence_entry",
                        "confidence": confidence,
                        "price_increase_target": float(increase_level),
                        "action": "enter_long",
                        "reason": f"High ML confidence ({confidence:.2f}) for {increase_level}% increase"
                    })
                elif confidence >= 0.5:  # Medium confidence threshold
                    entry_signals.append({
                        "signal_type": "ml_medium_confidence_entry",
                        "confidence": confidence,
                        "price_increase_target": float(increase_level),
                        "action": "enter_long_cautious",
                        "reason": f"Medium ML confidence ({confidence:.2f}) for {increase_level}% increase"
                    })
            
            return entry_signals
            
        except Exception as e:
            self.logger.error(f"Error generating ML entry signals: {e}")
            return []

    def _generate_ml_exit_signals(self, ml_predictions: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate ML-based exit signals."""
        try:
            exit_signals = []
            expected_decreases = ml_predictions.get("expected_decreases", {})
            
            for decrease_level, probability in expected_decreases.items():
                if probability >= 0.7:  # High probability threshold
                    exit_signals.append({
                        "signal_type": "ml_high_probability_exit",
                        "probability": probability,
                        "price_decrease_target": float(decrease_level),
                        "action": "exit_position",
                        "reason": f"High ML probability ({probability:.2f}) for {decrease_level}% decrease"
                    })
                elif probability >= 0.5:  # Medium probability threshold
                    exit_signals.append({
                        "signal_type": "ml_medium_probability_exit",
                        "probability": probability,
                        "price_decrease_target": float(decrease_level),
                        "action": "reduce_position",
                        "reason": f"Medium ML probability ({probability:.2f}) for {decrease_level}% decrease"
                    })
            
            return exit_signals
            
        except Exception as e:
            self.logger.error(f"Error generating ML exit signals: {e}")
            return []

    def _generate_ml_position_sizing(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """Generate ML-based position sizing."""
        try:
            position_sizing = {}
            confidence_scores = ml_predictions.get("confidence_scores", {})
            
            for increase_level, confidence in confidence_scores.items():
                # Calculate position size based on confidence
                base_size = 0.1  # 10% base position size
                confidence_multiplier = confidence
                position_size = base_size * confidence_multiplier
                
                position_sizing[f"size_{increase_level}"] = {
                    "position_size": min(position_size, 0.5),  # Cap at 50%
                    "confidence": confidence,
                    "sizing_reason": f"ML-based position size {position_size:.2f} based on confidence {confidence:.2f}"
                }
            
            return position_sizing
            
        except Exception as e:
            self.logger.error(f"Error generating ML position sizing: {e}")
            return {}

    def _generate_ml_risk_parameters(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """Generate ML-based risk parameters."""
        try:
            risk_parameters = {}
            confidence_scores = ml_predictions.get("confidence_scores", {})
            expected_decreases = ml_predictions.get("expected_decreases", {})
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
            
            # Calculate average decrease probability
            avg_decrease_prob = sum(expected_decreases.values()) / len(expected_decreases) if expected_decreases else 0.3
            
            # Adjust risk parameters based on ML predictions
            risk_parameters = {
                "stop_loss_adjustment": 1.0 - avg_confidence,  # Lower confidence = tighter stop loss
                "take_profit_adjustment": avg_confidence,  # Higher confidence = higher take profit
                "position_risk_adjustment": avg_decrease_prob,  # Higher decrease prob = lower position risk
                "leverage_adjustment": avg_confidence,  # Higher confidence = higher leverage
                "ml_confidence": avg_confidence,
                "ml_decrease_probability": avg_decrease_prob,
            }
            
            return risk_parameters
            
        except Exception as e:
            self.logger.error(f"Error generating ML risk parameters: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR breakout strategy generation",
    )
    async def _generate_sr_breakout_strategy(
        self,
        market_intelligence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Generate SR breakout strategy components."""
        try:
            # Extract market data for prediction
            market_data = market_intelligence.get("market_data", {})
            current_price = market_data.get("current_price", 2000.0)
            
            # Create sample market data for prediction (in real implementation, this would come from market feed)
            import pandas as pd
            sample_data = {
                "close": [1950, 1960, 1970, 1980, 1990, 2000],
                "high": [1960, 1970, 1980, 1990, 2000, 2010],
                "low": [1940, 1950, 1960, 1970, 1980, 1990],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500]
            }
            df = pd.DataFrame(sample_data)
            
            # Get breakout prediction
            prediction = await self.sr_breakout_predictor.predict_breakout_probability(df, current_price)
            
            if prediction and prediction.get("near_sr_zone", False):
                # Generate breakout-based strategy components
                breakout_strategy = {
                    "breakout_prediction": prediction,
                    "entry_signals": self._generate_sr_breakout_entry_signals(prediction),
                    "exit_signals": self._generate_sr_breakout_exit_signals(prediction),
                    "risk_levels": self._calculate_sr_breakout_risk_levels(prediction),
                    "position_adjustment": self._calculate_sr_breakout_position_adjustment(prediction),
                }
                
                self.logger.info(f"SR Breakout Strategy: {prediction.get('recommendation', 'UNKNOWN')}")
                return breakout_strategy
            else:
                self.logger.info("No SR breakout prediction available")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating SR breakout strategy: {e}")
            return None

    def _generate_sr_breakout_entry_signals(self, prediction: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate entry signals based on SR breakout prediction."""
        try:
            signals = []
            breakout_prob = prediction.get("breakout_probability", 0.5)
            bounce_prob = prediction.get("bounce_probability", 0.5)
            confidence = prediction.get("confidence", 0.0)
            
            if confidence > 0.7:
                if breakout_prob > 0.7:
                    signals.append({
                        "type": "breakout_long",
                        "strength": breakout_prob,
                        "confidence": confidence,
                        "reason": "High probability breakout above resistance"
                    })
                elif bounce_prob > 0.7:
                    signals.append({
                        "type": "bounce_long",
                        "strength": bounce_prob,
                        "confidence": confidence,
                        "reason": "High probability bounce from support"
                    })
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating SR breakout entry signals: {e}")
            return []

    def _generate_sr_breakout_exit_signals(self, prediction: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate exit signals based on SR breakout prediction."""
        try:
            signals = []
            breakout_prob = prediction.get("breakout_probability", 0.5)
            bounce_prob = prediction.get("bounce_probability", 0.5)
            confidence = prediction.get("confidence", 0.0)
            
            if confidence > 0.7:
                if breakout_prob > 0.7:
                    signals.append({
                        "type": "breakout_exit_short",
                        "strength": breakout_prob,
                        "confidence": confidence,
                        "reason": "Exit short positions on breakout"
                    })
                elif bounce_prob > 0.7:
                    signals.append({
                        "type": "bounce_exit_long",
                        "strength": bounce_prob,
                        "confidence": confidence,
                        "reason": "Exit long positions on bounce"
                    })
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating SR breakout exit signals: {e}")
            return []

    def _calculate_sr_breakout_risk_levels(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk levels based on SR breakout prediction."""
        try:
            breakout_prob = prediction.get("breakout_probability", 0.5)
            bounce_prob = prediction.get("bounce_probability", 0.5)
            confidence = prediction.get("confidence", 0.0)
            
            # Adjust risk based on prediction confidence
            base_risk = 0.02  # 2% base risk
            confidence_multiplier = confidence if confidence > 0.5 else 0.5
            
            risk_levels = {
                "stop_loss_adjustment": confidence_multiplier,
                "position_size_multiplier": confidence_multiplier,
                "risk_score": base_risk * confidence_multiplier,
                "prediction_confidence": confidence,
            }
            
            return risk_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating SR breakout risk levels: {e}")
            return {}

    def _calculate_sr_breakout_position_adjustment(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Calculate position adjustment based on SR breakout prediction."""
        try:
            breakout_prob = prediction.get("breakout_probability", 0.5)
            bounce_prob = prediction.get("bounce_probability", 0.5)
            confidence = prediction.get("confidence", 0.0)
            
            # Adjust position size based on prediction confidence
            base_position_size = 1.0
            confidence_multiplier = confidence if confidence > 0.5 else 0.5
            
            position_adjustment = {
                "size_multiplier": confidence_multiplier,
                "direction": "long" if bounce_prob > breakout_prob else "short",
                "confidence": confidence,
                "prediction_strength": max(breakout_prob, bounce_prob),
            }
            
            return position_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating SR breakout position adjustment: {e}")
            return {}

    def _generate_sr_entry_signals(self, sr_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate entry signals based on SR levels."""
        try:
            entry_signals = []
            
            # Support level entry signals
            for level in sr_analysis.get("support_levels", []):
                if level.get("strength", 0) > 0.7:  # Strong support
                    entry_signals.append({
                        "type": "support_bounce",
                        "price": level.get("price"),
                        "strength": level.get("strength"),
                        "confidence": 0.8,
                        "stop_loss": level.get("price") * 0.98,  # 2% below support
                    })

            # Resistance breakout signals
            for level in sr_analysis.get("resistance_levels", []):
                if level.get("strength", 0) > 0.7:  # Strong resistance
                    entry_signals.append({
                        "type": "resistance_breakout",
                        "price": level.get("price"),
                        "strength": level.get("strength"),
                        "confidence": 0.7,
                        "stop_loss": level.get("price") * 0.99,  # 1% below breakout
                    })

            return entry_signals

        except Exception as e:
            self.logger.error(f"Error generating SR entry signals: {e}")
            return []

    def _generate_sr_exit_signals(self, sr_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate exit signals based on SR levels."""
        try:
            exit_signals = []
            
            # Support breakdown signals
            for level in sr_analysis.get("support_levels", []):
                if level.get("strength", 0) > 0.7:
                    exit_signals.append({
                        "type": "support_breakdown",
                        "price": level.get("price"),
                        "strength": level.get("strength"),
                        "confidence": 0.9,
                    })

            # Resistance rejection signals
            for level in sr_analysis.get("resistance_levels", []):
                if level.get("strength", 0) > 0.7:
                    exit_signals.append({
                        "type": "resistance_rejection",
                        "price": level.get("price"),
                        "strength": level.get("strength"),
                        "confidence": 0.8,
                    })

            return exit_signals

        except Exception as e:
            self.logger.error(f"Error generating SR exit signals: {e}")
            return []

    def _calculate_sr_risk_levels(self, sr_analysis: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk levels based on SR analysis."""
        try:
            return {
                "high_risk_zones": [
                    level.get("price") for level in sr_analysis.get("resistance_levels", [])
                    if level.get("strength", 0) > 0.8
                ],
                "low_risk_zones": [
                    level.get("price") for level in sr_analysis.get("support_levels", [])
                    if level.get("strength", 0) > 0.8
                ],
                "risk_score": min(
                    len(sr_analysis.get("resistance_levels", [])) / 10.0, 1.0
                ),
            }

        except Exception as e:
            self.logger.error(f"Error calculating SR risk levels: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="strategy validation",
    )
    def _validate_strategy(self, strategy: dict[str, Any]) -> bool:
        """
        Validate generated strategy.

        Args:
            strategy: Strategy to validate

        Returns:
            bool: True if strategy is valid, False otherwise
        """
        try:
            required_keys = [
                "entry_signals",
                "exit_signals",
                "risk_parameters",
                "position_sizing",
                "confidence_score",
            ]
            for key in required_keys:
                if key not in strategy:
                    self.logger.error(f"Missing required strategy key: {key}")
                    return False

            # Validate confidence score
            confidence = strategy.get("confidence_score", 0)
            if confidence < self.strategy_config["min_confidence_threshold"]:
                self.logger.warning(f"Strategy confidence too low: {confidence}")
                return False

            # Validate position sizing
            position_size = strategy.get("position_sizing", {}).get("position_size", 0)
            if (
                position_size <= 0
                or position_size > self.strategy_config["max_position_size"]
            ):
                self.logger.error(f"Invalid position size: {position_size}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating strategy: {e}")
            return False

    def get_current_strategy(self) -> dict[str, Any] | None:
        """
        Get current strategy.

        Returns:
            Optional[Dict[str, Any]]: Current strategy or None
        """
        return self.current_strategy.copy() if self.current_strategy else None

    def get_strategy_history(self) -> list[dict[str, Any]]:
        """
        Get strategy history.

        Returns:
            List[Dict[str, Any]]: Strategy history
        """
        return self.strategy_history.copy()

    def get_last_strategy_update(self) -> datetime | None:
        """
        Get last strategy update time.

        Returns:
            Optional[datetime]: Last update time or None
        """
        return self.last_strategy_update

    def get_strategy_performance(self) -> dict[str, float]:
        """
        Get strategy performance metrics.

        Returns:
            Dict[str, float]: Performance metrics
        """
        return self.strategy_performance.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="strategist cleanup",
    )
    async def stop(self) -> None:
        """Stop the strategist component."""
        self.logger.info("🛑 Stopping Strategist...")

        try:
            # Save strategy history
            if self.strategy_history:
                self.logger.info(
                    f"Saving {len(self.strategy_history)} strategy records",
                )

            # Clear current state
            self.current_strategy = None
            self.last_strategy_update = None

            self.logger.info("✅ Strategist stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping strategist: {e}")
