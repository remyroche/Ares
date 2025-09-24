

from datetime import datetime
from typing import Any, TYPE_CHECKING

# Note: compat module has been refactored, using enhanced_error_handler instead
from ..utils.enhanced_error_handler import handle_errors_with_tracking
from ..utils.logger import system_logger
from ..utils.warning_symbols import failed, initialization_error, warning
from ..core.error_classes import ValidationError
from ..core.decorators import handles_errors
from ..utils.compat import handle_specific_errors
# Live trading utilities
from src.utils.model_manager import ModelManager
# Performance monitoring
from src.utils.performance_utils import PerformanceMonitor, global_monitor
from src.utils.unified_cache import cached
# Live trading validation
import pandas as pd

from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)
from .enhanced_regime_classifier import EnhancedRegimeClassifier

"""
Strategist module for trading strategy generation.

This module provides the Strategist class which is responsible for:
- Strategy Generation: Create trading strategies based on market analysis
- Market Analysis Integration: Combine analyst and tactician inputs
- Strategy History Management: Track and store strategy performance
"""

# Import Pydantic models and utilities
from .config import MarketIndicators, StrategistConfig, StrategyResult

from .utils import (
    CalculationError,
    PerformanceOptimizer,
    StrategyComponentExtractor,
    ValidationError,
    create_strategy_validator,
    log_error,
    validate_data_sufficiency,
    validate_required_columns,
)

if TYPE_CHECKING:
    from src.analyst.analyst import Analyst
    from src.tactician.tactician import Tactician

class Strategist:
    """
    Strategy-Level Strategist component responsible for:
    - Strategy Generation: Create trading strategies based on market analysis
    - Market Analysis Integration: Combine analyst and tactician inputs
    - Strategy History Management: Track and store strategy performance

    Note: Position sizing is handled by the Tactician component
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize strategist with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("Strategist")

        # Parse configuration using Pydantic
        strategist_config_dict = self.config.get("strategist", {})
        self.strategist_config = StrategistConfig(**strategist_config_dict)

        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(
            use_vectorized = self.strategist_config.use_vectorized_calculations,
            use_parallel = self.strategist_config.parallel_indicator_calculation,
            cache_ttl = self.strategist_config.cache_ttl,
        )

        # Component extractor for reducing complexity
        self.component_extractor = StrategyComponentExtractor()

        # Strategist state
        self.is_running: bool = False
        self.strategy_results: dict[str, Any] = {}
        self.strategy_history: list[dict[str, Any]] = []
        self.current_strategy: dict[str, Any] = {}

        # Component references (will be set during initialization)
        self.analyst: Analyst | None = None
        self.tactician: Tactician | None = None
        
        # Enhanced regime classifier (lazily imported during initialize)
        self.regime_classifier: "EnhancedRegimeClassifier" | None = None
        self.enable_regime_detection = self.strategist_config.dict().get("enable_regime_detection", True)
        
        # Live trading utilities
        self.model_manager: ModelManager | None = None
        self.selected_model: str | None = None
        self.model_cache: dict[str, Any] = {}
        
        # Performance monitoring for live trading
        self.performance_monitor: PerformanceMonitor | None = None
        self.global_monitor = global_monitor
        self.strategy_cache: dict[str, Any] = {}

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid strategist configuration"),
            AttributeError: (False, "Missing required strategist parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return = False,
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

            # Configuration is already validated by Pydantic
            self.logger.info("✅ Configuration validated successfully")

            # Initialize strategy components
            await self._initialize_strategy_components()
            
            # Initialize regime classifier if enabled
            if self.enable_regime_detection:
                regime_config = self.config.get("strategist", {}).get("regime_classifier", {})
                try:

                    self.regime_classifier = EnhancedRegimeClassifier(regime_config)
                    await self.regime_classifier.initialize()
                    self.logger.info("✅ Enhanced regime classifier initialized")
                except Exception as e:
                    self.logger.warning(
                        f"Regime classifier unavailable or failed to initialize ({e}); disabling regime detection"
                    )
                    self.enable_regime_detection = False
                    self.regime_classifier = None
            
            # Initialize live trading utilities
            await self._initialize_live_trading_utilities()
            
            # Initialize performance monitoring
            await self._initialize_performance_monitoring()

            self.logger.info("✅ Strategist initialized successfully")
            return True

        except Exception as e:
            log_error(self.logger, "❌ Strategist initialization failed", e)
            return False

    async def _initialize_strategy_components(self) -> None:
        """Initialize strategy components."""
        try:
            # Initialize risk management
            if self.strategist_config.enable_risk_management:
                self.logger.info("Initializing risk management components...")

            # Position sizing is handled by the Tactician component
            self.logger.info("✅ Strategy components initialized successfully")

        except Exception as e:
            log_error(self.logger, "Error initializing strategy components", e)
            raise

    @handle_specific_errors(
        error_handlers={
            ValidationError: (None, "Invalid market data for strategy generation"),
            CalculationError: (None, "Error in market calculations"),
            Exception: (None, "Unexpected error in strategy generation"),
        },
        default_return = None,
        context="strategy generation",
    )
    @create_strategy_validator(min_confidence = 0.0, max_confidence = 1.0)
    @cached(ttl=120, key_func=lambda self, market_data, current_price, analysis_results: f"strategy_{current_price}_{hash(str(market_data.tail(10).values.tolist()))}")
    @global_monitor.track_function
    async def generate_strategy(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analysis_results: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate trading strategy based on market data and analysis results.

        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            analysis_results: Results from market analysis (Step 1)

        Returns:
            Generated strategy or None if failed
        """
        try:
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_timer("strategy_generation")
            
            # Validate market data
            self._validate_market_data(market_data)

            self.logger.info("🎯 Generating trading strategy...")
            tprint("🎯 Generating trading strategy...")

            # Extract market indicators using performance optimizer
            market_indicators = await self._extract_market_indicators_optimized(
                market_data,
                current_price,
            )
            
            # Detect market regime if enabled
            regime = "MODERATE_BULL"  # Default
            regime_confidence = 0.5
            regime_metadata = {}
            regime_params = {}
            
            if self.enable_regime_detection and self.regime_classifier:
                regime, regime_confidence, regime_metadata = await self.regime_classifier.predict_regime(market_data)
                regime_params = self.regime_classifier.get_regime_strategy_params(regime)
                self.logger.info(f"Detected regime: {regime} (confidence: {regime_confidence:.2%})")

            # Generate base strategy
            base_strategy = self._generate_base_strategy_simplified(
                market_indicators,
                current_price,
            )
            
            # Apply regime-specific adjustments
            if self.enable_regime_detection:
                base_strategy = self._apply_regime_adjustments(
                    base_strategy,
                    regime,
                    regime_confidence,
                    regime_params,
                    regime_metadata
                )

            # Integrate analysis results if available
            if analysis_results:
                base_strategy = self._integrate_analysis_results_simplified(
                    base_strategy,
                    analysis_results,
                )

            # Apply risk management
            if self.strategist_config.enable_risk_management:
                base_strategy = self._apply_risk_management_simplified(
                    base_strategy,
                    current_price,
                )

            # Store results
            self._store_strategy_results(base_strategy)

            # End performance monitoring
            if self.performance_monitor:
                execution_time = self.performance_monitor.end_timer("strategy_generation")
                self.logger.info(f"Strategy generation completed in {execution_time:.3f}s")
                tprint(f"Strategy generation completed in {execution_time:.3f}s")
            
            self.logger.info(f"✅ Strategy generated: {base_strategy.get('direction', 'UNKNOWN')} with confidence {base_strategy.get('confidence', 0.0):.3f}")
            tprint(f"✅ Strategy generated: {base_strategy.get('direction', 'UNKNOWN')} with confidence {base_strategy.get('confidence', 0.0):.3f}")
            return base_strategy

        except ValidationError as e:
            error_msg = f"Validation error in strategy generation: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            log_error(self.logger, "Validation error in strategy generation", e)
            
            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("strategy_generation")
            
            return None
        except Exception as e:
            error_msg = f"Error generating strategy: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            log_error(self.logger, "Error generating strategy", e)
            
            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("strategy_generation")
            
            return None

    def _validate_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Validate market data for strategy generation.

        Raises:
            ValidationError: If validation fails
        """
        required_columns = ["close", "volume", "timestamp"]
        validate_required_columns(market_data, required_columns)
        validate_data_sufficiency(market_data, min_rows = 100)

    async def _extract_market_indicators_optimized(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> MarketIndicators:
        """
        Extract market indicators with performance optimization.

        Args:
            market_data: Market data DataFrame
            current_price: Current price

        Returns:
            MarketIndicators object with calculated values
        """
        try:
            # Use performance optimizer for parallel calculation
            config_dict = self.strategist_config.technical_indicator_thresholds.dict()
            indicators = await self.optimizer.calculate_indicators_parallel(
                market_data["close"],
                market_data["volume"],
                config_dict,
            )

            # Calculate additional indicators
            price_change_percent = (
                (current_price - market_data["close"].iloc[-2])
                / market_data["close"].iloc[-2]
                * 100
            )

            sma_trend = (
                "BULLISH"
                if indicators.get("sma_fast", 0) > indicators.get("sma_slow", 0)
                else "BEARISH"
            )

            return MarketIndicators(
                rsi = indicators.get("rsi"),
                sma_fast = indicators.get("sma_fast"),
                sma_slow = indicators.get("sma_slow"),
                volume_ratio = indicators.get("volume_ratio"),
                volatility = indicators.get("volatility"),
                price_change_percent = price_change_percent,
                sma_trend = sma_trend,
            )

        except Exception as e:
            msg = f"Failed to extract market indicators: {e}"
            raise CalculationError(msg)

    def _generate_base_strategy_simplified(
        self,
        indicators: MarketIndicators,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Generate base strategy with simplified logic.

        Args:
            indicators: Calculated market indicators
            current_price: Current price

        Returns:
            Base strategy dictionary
        """
        strategy = StrategyResult(
            direction="HOLD",
            confidence = 0.5,
            reasoning=[],
            timestamp = datetime.now().isoformat(),
        ).dict()

        # RSI-based signals
        if indicators.rsi is not None:
            if (
                indicators.rsi
                < self.strategist_config.technical_indicator_thresholds.rsi_oversold
            ):
                strategy["direction"] = "BUY"
                strategy["confidence"] += 0.2
                strategy["reasoning"].append(f"RSI oversold ({indicators.rsi:.2f})")
            elif (
                indicators.rsi
                > self.strategist_config.technical_indicator_thresholds.rsi_overbought
            ):
                strategy["direction"] = "SELL"
                strategy["confidence"] += 0.2
                strategy["reasoning"].append(f"RSI overbought ({indicators.rsi:.2f})")

        # SMA crossover signals
        if indicators.sma_trend == "BULLISH" and strategy["direction"] != "SELL":
            strategy["direction"] = "BUY"
            strategy["confidence"] += 0.15
            strategy["reasoning"].append("Bullish SMA crossover")
        elif indicators.sma_trend == "BEARISH" and strategy["direction"] != "BUY":
            strategy["direction"] = "SELL"
            strategy["confidence"] += 0.15
            strategy["reasoning"].append("Bearish SMA crossover")

        # Volume confirmation
        if indicators.volume_ratio is not None:
            if (
                indicators.volume_ratio
                > self.strategist_config.technical_indicator_thresholds.volume_ratio_high
            ):
                strategy["confidence"] += 0.1
                strategy["reasoning"].append("High volume confirmation")

        # Normalize confidence
        strategy["confidence"] = min(strategy["confidence"], 1.0)

        return strategy

    def _integrate_analysis_results_simplified(
        self,
        strategy: dict[str, Any],
        analysis_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Integrate analysis results with simplified, modular approach.

        Args:
            strategy: Base strategy
            analysis_results: Analysis results to integrate

        Returns:
            Updated strategy
        """
        # Extract market health component
        health_component = self.component_extractor.extract_market_health(
            analysis_results
        )
        if health_component:
            strategy["market_health_score"] = health_component.get("health_score")
            if "health_impact" in health_component:
                strategy["confidence"] = (
                    strategy["confidence"] + health_component["health_impact"]
                ) / 2
            if health_component.get("reasoning"):
                strategy["reasoning"].append(health_component["reasoning"])

        # Extract liquidation risk component
        risk_component = self.component_extractor.extract_liquidation_risk(
            analysis_results
        )
        if risk_component:
            strategy["liquidation_risk"] = risk_component.get("risk_level")
            strategy["confidence"] *= risk_component.get("confidence_multiplier", 1.0)
            if risk_component.get("reasoning"):
                strategy["reasoning"].append(risk_component["reasoning"])

        # Extract trading decision component
        decision_component = self.component_extractor.extract_trading_decision(
            analysis_results
        )
        if decision_component:
            strategy.update(
                {
                    "dual_model_direction": decision_component.get(
                        "dual_model_direction"
                    ),
                    "dual_model_confidence": decision_component.get(
                        "dual_model_confidence"
                    ),
                    "direction": decision_component.get(
                        "direction", strategy["direction"]
                    ),
                    "confidence": decision_component.get(
                        "confidence", strategy["confidence"]
                    ),
                }
            )
            if decision_component.get("reasoning"):
                strategy["reasoning"].append(decision_component["reasoning"])

        return strategy

    def _apply_risk_management_simplified(
        self,
        strategy: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any]:
        """
        Apply risk management with simplified logic.

        Args:
            strategy: Strategy to apply risk management to
            current_price: Current price

        Returns:
            Strategy with risk management applied
        """
        if strategy["direction"] == "HOLD":
            return strategy

        # Calculate stop loss and take profit based on direction
        risk_reward_ratio = 2.0  # 1:2 risk-reward ratio
        risk_percentage = 0.02  # 2% risk per trade

        if strategy["direction"] == "BUY":
            strategy["stop_loss"] = current_price * (1 - risk_percentage)
            strategy["take_profit"] = current_price * (
                1 + risk_percentage * risk_reward_ratio
            )
            strategy["reasoning"].append(
                f"Risk management: SL={strategy['stop_loss']:.2f}, TP={strategy['take_profit']:.2f}"
            )
        elif strategy["direction"] == "SELL":
            strategy["stop_loss"] = current_price * (1 + risk_percentage)
            strategy["take_profit"] = current_price * (
                1 - risk_percentage * risk_reward_ratio
            )
            strategy["reasoning"].append(
                f"Risk management: SL={strategy['stop_loss']:.2f}, TP={strategy['take_profit']:.2f}"
            )

        # Reduce confidence if it's below threshold
        if strategy["confidence"] < self.strategist_config.min_confidence_threshold:
            strategy["direction"] = "HOLD"
            strategy["reasoning"].append(
                f"Confidence below threshold ({self.strategist_config.min_confidence_threshold})"
            )

        return strategy

    def _store_strategy_results(self, strategy: dict[str, Any]) -> None:
        """Store strategy results with history management."""
        try:
            # Update current strategy
            self.current_strategy = strategy.copy()
            self.strategy_results = strategy.copy()

            # Add to history
            self.strategy_history.append(strategy.copy())

            # Maintain history size limit
            if len(self.strategy_history) > self.strategist_config.max_strategy_history:
                self.strategy_history.pop(0)

            self.logger.info(
                f"Strategy stored: {strategy['direction']} with confidence {strategy['confidence']:.3f}"
            )

        except Exception as e:
            log_error(self.logger, "Error storing strategy results", e)

    def get_strategy_results(self) -> dict[str, Any]:
        """Get the current strategy results."""
        return self.strategy_results.copy()

    def get_current_strategy(self) -> dict[str, Any]:
        """Get the current active strategy."""
        return self.current_strategy.copy()

    def get_strategy_history(self) -> list[dict[str, Any]]:
        """Get strategy history."""
        return self.strategy_history.copy()
    
    def _apply_regime_adjustments(
        self,
        strategy: dict[str, Any],
        regime: str,
        regime_confidence: float,
        regime_params: dict[str, Any],
        regime_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply regime-specific adjustments to strategy.
        
        Args:
            strategy: Base strategy
            regime: Detected market regime
            regime_confidence: Confidence in regime detection
            regime_params: Regime-specific parameters
            regime_metadata: Additional regime metadata
            
        Returns:
            Strategy adjusted for market regime
        """
        try:
            # Add regime information to strategy
            strategy["regime"] = regime
            strategy["regime_confidence"] = regime_confidence
            strategy["regime_metadata"] = regime_metadata
            
            # Adjust confidence based on regime alignment
            if regime in ["STRONG_BULL", "MODERATE_BULL"] and strategy["direction"] == "BUY":
                strategy["confidence"] *= regime_params.get("momentum_weight", 0.6)
                strategy["reasoning"].append(f"Bullish regime alignment ({regime})")
            elif regime in ["STRONG_BEAR", "MODERATE_BEAR"] and strategy["direction"] == "SELL":
                strategy["confidence"] *= regime_params.get("momentum_weight", 0.6)
                strategy["reasoning"].append(f"Bearish regime alignment ({regime})")
            elif regime in ["RANGING_HIGH", "RANGING_LOW"]:
                # Mean reversion in ranging markets
                if (regime == "RANGING_HIGH" and strategy["direction"] == "SELL") or \
                   (regime == "RANGING_LOW" and strategy["direction"] == "BUY"):
                    strategy["confidence"] *= regime_params.get("mean_reversion_weight", 0.7)
                    strategy["reasoning"].append(f"Mean reversion in {regime}")
                else:
                    strategy["confidence"] *= 0.8  # Reduce confidence for trend following in ranging markets
            elif regime in ["BREAKOUT_UP", "BREAKOUT_DOWN"]:
                # Favor breakout direction
                if (regime == "BREAKOUT_UP" and strategy["direction"] == "BUY") or \
                   (regime == "BREAKOUT_DOWN" and strategy["direction"] == "SELL"):
                    strategy["confidence"] *= 1.2
                    strategy["reasoning"].append(f"Breakout confirmation ({regime})")
            
            # Apply regime-specific thresholds
            strategy["entry_confidence_threshold"] = regime_params.get("entry_confidence_threshold", 0.65)
            strategy["position_size_multiplier"] = regime_params.get("position_size_multiplier", 1.0)
            strategy["stop_loss_multiplier"] = regime_params.get("stop_loss_multiplier", 1.0)
            strategy["take_profit_multiplier"] = regime_params.get("take_profit_multiplier", 1.2)
            
            # Ensure confidence stays within bounds
            strategy["confidence"] = max(0.0, min(1.0, strategy["confidence"]))
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Failed to apply regime adjustments: {e}")
            return strategy

    @handle_errors_with_tracking(
        context="live trading utilities initialization",
        log_level="INFO",
        print_errors=True
    )
    async def _initialize_live_trading_utilities(self) -> bool:
        """Initialize live trading utilities."""
        try:
            self.logger.info("Initializing live trading utilities...")
            tprint("Initializing live trading utilities...")
            
            # Initialize Model Manager for model selection and loading
            self.model_manager = ModelManager()
            self.logger.info("✅ Model Manager initialized")
            tprint("✅ Model Manager initialized")
            
            # Set default model selection for strategy generation (single model trained on various conditions)
            # The strategist now includes regime classification functionality
            self.selected_model = "strategist_regime_classifier"
            self.logger.info(f"✅ Default model selected: {self.selected_model}")
            tprint(f"✅ Default model selected: {self.selected_model}")
            
            # Initialize caches
            self.model_cache = {}
            self.strategy_cache = {}
            self.logger.info("✅ Model and strategy caches initialized")
            tprint("✅ Model and strategy caches initialized")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing live trading utilities: {e}")
            tprint(f"❌ Error initializing live trading utilities: {e}")
            return False

    @handle_errors_with_tracking(
        context="HMM regime classification",
        log_level="INFO",
        print_errors=True
    )
    async def classify_hmm_regime(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Classify HMM regime using the strategist's regime classifier model.
        
        Args:
            market_data: Market data for regime classification
            
        Returns:
            dict: Regime classification results
        """
        if not self.model_manager or not self.selected_model:
            error_msg = "Model Manager or selected model not available"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}
        
        try:
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_timer("regime_classification")
            
            self.logger.info("Classifying HMM regime...")
            tprint("Classifying HMM regime...")
            
            # Get model from cache or load it
            model = self.model_cache.get(self.selected_model)
            if not model:
                model = await self.model_manager.load_model(self.selected_model)
                if model:
                    self.model_cache[self.selected_model] = model
                else:
                    error_msg = f"Failed to load regime classifier model: {self.selected_model}"
                    self.logger.error(error_msg)
                    tprint(f"❌ {error_msg}")
                    return {"error": error_msg}
            
            # Get regime classification
            regime_result = await self.model_manager.get_prediction(model, market_data)
            
            # End performance monitoring
            if self.performance_monitor:
                execution_time = self.performance_monitor.end_timer("regime_classification")
                self.logger.info(f"Regime classification completed in {execution_time:.3f}s")
                tprint(f"Regime classification completed in {execution_time:.3f}s")
            
            self.logger.info(f"✅ HMM regime classified: {regime_result.get('regime', 'UNKNOWN')}")
            tprint(f"✅ HMM regime classified: {regime_result.get('regime', 'UNKNOWN')}")
            return regime_result
            
        except Exception as e:
            error_msg = f"Error classifying HMM regime: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            
            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("regime_classification")
            
            return {"error": error_msg}

    @handle_errors_with_tracking(
        context="HMM regime-based strategy coordination",
        log_level="INFO",
        print_errors=True
    )
    async def coordinate_strategy_with_hmm_regime(self, hmm_regime: str, regime_confidence: float) -> dict[str, Any]:
        """
        Coordinate strategy generation based on HMM regime detection.
        
        Args:
            hmm_regime: Detected HMM regime (15-25 possible regimes)
            regime_confidence: Confidence in the regime detection
            
        Returns:
            dict: Strategy coordination results and regime-specific parameters
        """
        if not self.model_manager or not self.selected_model:
            error_msg = "Model Manager or selected model not available"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}
        
        try:
            self.logger.info(f"Coordinating strategy with HMM regime: {hmm_regime} (confidence: {regime_confidence:.3f})")
            tprint(f"Coordinating strategy with HMM regime: {hmm_regime} (confidence: {regime_confidence:.3f})")
            
            # Get the single model (trained on various market conditions)
            model = self.model_cache.get(self.selected_model)
            if not model:
                error_msg = f"Model {self.selected_model} not loaded in cache"
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return {"error": error_msg}
            
            # Configure regime-specific parameters for strategy generation
            regime_config = {
                "hmm_regime": hmm_regime,
                "regime_confidence": regime_confidence,
                "model_name": self.selected_model,
                "regime_parameters": self._get_optimized_strategy_parameters(hmm_regime, regime_confidence)
            }
            
            self.logger.info(f"✅ Strategy coordination with HMM regime completed: {hmm_regime}")
            tprint(f"✅ Strategy coordination with HMM regime completed: {hmm_regime}")
            return regime_config
            
        except Exception as e:
            error_msg = f"Error coordinating strategy with HMM regime: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

    def _get_optimized_strategy_parameters(self, hmm_regime: str, regime_confidence: float) -> dict[str, Any]:
        """
        Get optimized strategy parameters for HMM regime from training optimization.
        
        Args:
            hmm_regime: Detected HMM regime (15-25 possible regimes)
            regime_confidence: Confidence in regime detection
            
        Returns:
            dict: Optimized strategy parameters for the regime
        """
        try:
            # Load optimized parameters from training (final_parameters_optimization.py)
            # These parameters are optimized during training and stored in the model artifacts
            optimized_params = self._load_optimized_strategy_parameters_for_regime(hmm_regime)
            
            if optimized_params:
                # Apply confidence-based adjustments
                confidence_adjustment = 0.8 + (regime_confidence * 0.4)  # 0.8 to 1.2 range
                
                adjusted_params = {}
                for param_name, param_value in optimized_params.items():
                    if param_name in ["strategy_aggressiveness", "risk_tolerance"]:
                        # Higher confidence = more aggressive strategy
                        adjusted_params[param_name] = param_value * confidence_adjustment
                    elif param_name in ["trend_following_weight"]:
                        # Higher confidence = more trend following
                        adjusted_params[param_name] = param_value * confidence_adjustment
                    else:
                        adjusted_params[param_name] = param_value
                
                return adjusted_params
            else:
                # Fallback to default parameters if optimization not available
                return self._get_default_strategy_parameters(hmm_regime, regime_confidence)
                
        except Exception as e:
            self.logger.error(f"Error getting optimized strategy parameters: {e}")
            return self._get_default_strategy_parameters(hmm_regime, regime_confidence)

    def _load_optimized_strategy_parameters_for_regime(self, hmm_regime: str) -> dict[str, Any] | None:
        """
        Load optimized strategy parameters for a specific regime from training artifacts.
        
        Args:
            hmm_regime: HMM regime identifier
            
        Returns:
            dict: Optimized parameters or None if not found
        """
        try:
            # This would load from the optimized parameters saved during training
            # The parameters are optimized in final_parameters_optimization.py
            # and stored in model artifacts
            
            # For now, return None to use fallback parameters
            # In production, this would load from:
            # - Model artifacts
            # - Optimization results from final_parameters_optimization.py
            # - Regime-specific parameter files
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading optimized strategy parameters for regime {hmm_regime}: {e}")
            return None

    def _get_default_strategy_parameters(self, hmm_regime: str, regime_confidence: float) -> dict[str, Any]:
        """
        Get default strategy parameters as fallback.
        
        Args:
            hmm_regime: HMM regime identifier
            regime_confidence: Confidence in regime detection
            
        Returns:
            dict: Default parameters for the regime
        """
        # Base parameters that work across all regimes
        base_params = {
            "strategy_aggressiveness": 1.0,
            "risk_tolerance": 0.6,
            "trend_following_weight": 0.5,
            "regime_weight": 0.3
        }
        
        # Apply confidence-based adjustments
        confidence_adjustment = 0.8 + (regime_confidence * 0.4)
        
        adjusted_params = {}
        for param_name, param_value in base_params.items():
            if param_name in ["strategy_aggressiveness", "risk_tolerance", "trend_following_weight"]:
                adjusted_params[param_name] = param_value * confidence_adjustment
            else:
                adjusted_params[param_name] = param_value
        
        return adjusted_params

    @handles_errors(Exception, fallback = False)
    async def _initialize_performance_monitoring(self) -> bool:
        """Initialize performance monitoring."""
        try:
            self.logger.info("Initializing performance monitoring...")
            
            # Initialize Performance Monitor
            self.performance_monitor = PerformanceMonitor()
            self.logger.info("✅ Performance Monitor initialized")
            
            # Enable global monitoring
            self.global_monitor.enable()
            self.logger.info("✅ Global monitoring enabled")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing performance monitoring: {e}")
            return False

    @handle_errors_with_tracking(
        context="strategist cleanup",
        log_level="INFO",
        print_errors=True
    )
    async def stop(self) -> bool:
        """Stop the strategist component."""
        try:
            self.logger.info("Stopping Strategist...")
            tprint("Stopping Strategist...")
            self.is_running = False

            # Cleanup optimizer resources with enhanced error handling
            if hasattr(self, "optimizer") and self.optimizer._executor:
                try:
                    self.optimizer._executor.shutdown(wait = True)
                    self.logger.info("✅ Optimizer executor shutdown successfully")
                    tprint("✅ Optimizer executor shutdown successfully")
                except Exception as e:
                    self.logger.error(f"❌ Error shutting down optimizer executor: {e}")
                    tprint(f"❌ Error shutting down optimizer executor: {e}")

            # Clean up live trading utilities
            if self.model_manager:
                try:
                    # Clear model cache
                    self.model_cache.clear()
                    self.strategy_cache.clear()
                    self.logger.info("✅ Model and strategy caches cleared")
                    tprint("✅ Model and strategy caches cleared")
                except Exception as e:
                    self.logger.error(f"❌ Error cleaning up model caches: {e}")
                    tprint(f"❌ Error cleaning up model caches: {e}")

            if self.performance_monitor:
                try:
                    self.performance_monitor.stop()
                    self.logger.info("✅ Performance monitor stopped")
                    tprint("✅ Performance monitor stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping performance monitor: {e}")
                    tprint(f"❌ Error stopping performance monitor: {e}")

            self.logger.info("✅ Strategist stopped successfully")
            tprint("✅ Strategist stopped successfully")
            return True

        except Exception as e:
            error_msg = f"❌ Failed to stop Strategist: {e}"
            self.logger.error(error_msg)
            tprint(error_msg)
            log_error(self.logger, "❌ Failed to stop Strategist", e)
            return False
