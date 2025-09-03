"""Dynamic Weighter Module.

This module provides dynamic weighting strategies for ensemble models,
including performance-based, risk-based, adaptive, momentum, and volatility
weighting methods. It supports regime-aware and uncertainty-aware weighting
for optimal model combination.
"""

from collections import deque
from datetime import datetime
from typing import Any

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class DynamicWeighter:
    """
    Dynamic Weighter with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize dynamic weighter with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("DynamicWeighter")

        # Dynamic weighter state
        self.is_weighting: bool = False
        self.weighting_results: dict[str, Any] = {}
        self.weighting_history: list[dict[str, Any]] = []

        # Configuration
        self.weighter_config: dict[str, Any] = self.config.get("dynamic_weighter", {})
        self.weighting_interval: int = self.weighter_config.get(
            "weighting_interval",
            3600,
        )
        self.max_weighting_history: int = self.weighter_config.get(
            "max_weighting_history",
            100,
        )
        self.enable_performance_weighting: bool = self.weighter_config.get("enable_performance_weighting", True)
        self.enable_risk_weighting: bool = self.weighter_config.get("enable_risk_weighting", True)
        self.enable_adaptive_weighting: bool = self.weighter_config.get("enable_adaptive_weighting", True)

        # Enhanced ensemble weighting configuration
        self.enable_online_learning: bool = self.weighter_config.get("enable_online_learning", True)
        self.enable_regime_awareness: bool = self.weighter_config.get("enable_regime_awareness", True)
        self.enable_uncertainty_weighting: bool = self.weighter_config.get("enable_uncertainty_weighting", True)
        self.learning_rate: float = self.weighter_config.get("learning_rate", 0.01)
        self.performance_window: int = self.weighter_config.get("performance_window", 100)

        # Ensemble weighting state
        self.model_weights: dict[str, float] = {}
        self.model_performances: dict[str, deque] = {}  # Using deque for O(1) append/popleft
        self.regime_performances: dict[str, dict[str, float]] = {}
        self.uncertainty_metrics: dict[str, float] = {}

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid dynamic weighter configuration"),
            AttributeError: (False, "Missing required dynamic weighter parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="dynamic weighter initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize dynamic weighter with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Dynamic Weighter...")

            # Load dynamic weighter configuration
            await self._load_weighter_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for dynamic weighter")
                return False

            # Initialize dynamic weighter modules
            await self._initialize_weighter_modules()

            self.logger.info(
                "✅ Dynamic Weighter initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Dynamic Weighter initialization failed: {e}")
            return False

    @handles_errors(fallback=None)
    async def _load_weighter_configuration(self) -> None:
        """Load dynamic weighter configuration."""
        try:
            # Set default weighter parameters
            self.weighter_config.setdefault("weighting_interval", 3600)
            self.weighter_config.setdefault("max_weighting_history", 100)
            self.weighter_config.setdefault("enable_performance_weighting", True)
            self.weighter_config.setdefault("enable_risk_weighting", True)
            self.weighter_config.setdefault("enable_adaptive_weighting", True)
            self.weighter_config.setdefault("enable_momentum_weighting", True)
            self.weighter_config.setdefault("enable_volatility_weighting", True)

            # Update configuration
            self.weighting_interval = self.weighter_config["weighting_interval"]
            self.max_weighting_history = self.weighter_config["max_weighting_history"]
            self.enable_performance_weighting = self.weighter_config["enable_performance_weighting"]
            self.enable_risk_weighting = self.weighter_config["enable_risk_weighting"]
            self.enable_adaptive_weighting = self.weighter_config["enable_adaptive_weighting"]

            self.logger.info("Dynamic weighter configuration loaded successfully")

        except Exception as e:
            self.logger.exception(f"Error loading weighter configuration: {e}")

    @handles_errors(fallback=False)
    def _validate_configuration(self) -> bool:
        """
        Validate dynamic weighter configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate weighting interval
            if self.weighting_interval <= 0:
                self.logger.error("Invalid weighting interval")
                return False

            # Validate max weighting history
            if self.max_weighting_history <= 0:
                self.logger.error("Invalid max weighting history")
                return False

            # Validate that at least one weighting type is enabled
            if not any(
                [
                    self.enable_performance_weighting,
                    self.enable_risk_weighting,
                    self.enable_adaptive_weighting,
                    self.weighter_config.get("enable_momentum_weighting", True),
                    self.weighter_config.get("enable_volatility_weighting", True),
                ],
            ):
                self.logger.error("At least one weighting type must be enabled")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.exception(f"Error validating configuration: {e}")
            return False

    @handles_errors(fallback=None)
    async def _initialize_weighter_modules(self) -> None:
        """Initialize dynamic weighter modules."""
        try:
            # Initialize performance weighting module
            if self.enable_performance_weighting:
                await self._initialize_performance_weighting()

            # Initialize risk weighting module
            if self.enable_risk_weighting:
                await self._initialize_risk_weighting()

            # Initialize adaptive weighting module
            if self.enable_adaptive_weighting:
                await self._initialize_adaptive_weighting()

            # Initialize momentum weighting module
            if self.weighter_config.get("enable_momentum_weighting", True):
                await self._initialize_momentum_weighting()

            # Initialize volatility weighting module
            if self.weighter_config.get("enable_volatility_weighting", True):
                await self._initialize_volatility_weighting()

            self.logger.info("Dynamic weighter modules initialized successfully")

        except Exception as e:
            self.logger.exception(f"Error initializing weighter modules: {e}")

    @handles_errors(fallback=None)
    async def _initialize_performance_weighting(self) -> None:
        """Initialize performance weighting components."""
        try:
            self.performance_weighting_components = {
                "return_based_weighting": True,
                "sharpe_based_weighting": True,
                "sortino_based_weighting": True,
                "calmar_based_weighting": True,
            }

            self.logger.info("Performance weighting components initialized")

        except Exception as e:
            self.logger.exception(f"Error initializing performance weighting: {e}")

    @handles_errors(fallback=None)
    async def _initialize_risk_weighting(self) -> None:
        """Initialize risk weighting components."""
        try:
            self.risk_weighting_components = {
                "var_based_weighting": True,
                "volatility_based_weighting": True,
                "drawdown_based_weighting": True,
                "correlation_based_weighting": True,
            }

            self.logger.info("Risk weighting components initialized")

        except Exception as e:
            self.logger.exception(f"Error initializing risk weighting: {e}")

    @handles_errors(fallback=None)
    async def _initialize_adaptive_weighting(self) -> None:
        """Initialize adaptive weighting components."""
        try:
            self.adaptive_weighting_components = {
                "market_regime_weighting": True,
                "regime_detection": True,
                "adaptive_learning": True,
                "dynamic_adjustment": True,
            }

            self.logger.info("Adaptive weighting components initialized")

        except Exception as e:
            self.logger.exception(f"Error initializing adaptive weighting: {e}")

    @handles_errors(fallback=None)
    async def _initialize_momentum_weighting(self) -> None:
        """Initialize momentum weighting components."""
        try:
            self.momentum_weighting_components = {
                "price_momentum_weighting": True,
                "volume_momentum_weighting": True,
                "momentum_breakout_weighting": True,
                "momentum_reversal_weighting": True,
            }

            self.logger.info("Momentum weighting components initialized")

        except Exception as e:
            self.logger.exception(f"Error initializing momentum weighting: {e}")

    @handles_errors(fallback=None)
    async def _initialize_volatility_weighting(self) -> None:
        """Initialize volatility weighting components."""
        try:
            self.volatility_weighting_components = {
                "realized_volatility_weighting": True,
                "implied_volatility_weighting": True,
                "volatility_regime_weighting": True,
                "volatility_forecast_weighting": True,
            }

            self.logger.info("Volatility weighting components initialized")

        except Exception as e:
            self.logger.exception(f"Error initializing volatility weighting: {e}")

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid weighting parameters"),
            AttributeError: (False, "Missing weighting components"),
            KeyError: (False, "Missing required weighting data"),
        },
        default_return=False,
        context="dynamic weighting execution",
    )
    async def execute_weighting(self, weighting_input: dict[str, Any]) -> bool:
        """
        Execute dynamic weighting with comprehensive error handling.

        Args:
            weighting_input: Input data for weighting calculation

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Executing Dynamic Weighting...")

            # Validate weighting inputs
            if not self._validate_weighting_inputs(weighting_input):
                self.logger.error("Invalid weighting inputs")
                return False

            # Set weighting state
            self.is_weighting = True

            # Perform performance weighting
            performance_results = await self._perform_performance_weighting(weighting_input)
            self.weighting_results["performance_weighting"] = performance_results

            # Perform risk weighting
            risk_results = await self._perform_risk_weighting(weighting_input)
            self.weighting_results["risk_weighting"] = risk_results

            # Perform adaptive weighting
            adaptive_results = await self._perform_adaptive_weighting(weighting_input)
            self.weighting_results["adaptive_weighting"] = adaptive_results

            # Perform momentum weighting
            momentum_results = await self._perform_momentum_weighting(weighting_input)
            self.weighting_results["momentum_weighting"] = momentum_results

            # Perform volatility weighting
            volatility_results = await self._perform_volatility_weighting(weighting_input)
            self.weighting_results["volatility_weighting"] = volatility_results

            # Update weighting history
            self._update_weighting_history()

            self.is_weighting = False
            self.logger.info("✅ Dynamic Weighting completed successfully")
            return True

        except Exception as e:
            self.is_weighting = False
            self.logger.exception(f"❌ Dynamic Weighting failed: {e}")
            return False

    @handles_errors(fallback=False)
    def _validate_weighting_inputs(self, weighting_input: dict[str, Any]) -> bool:
        """
        Validate weighting inputs.

        Args:
            weighting_input: Input data for validation

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not isinstance(weighting_input, dict):
                self.logger.error("Weighting input must be a dictionary")
                return False

            required_fields = ["weighting_type", "data_source", "timestamp"]
            for field in required_fields:
                if field not in weighting_input:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            self.logger.info("Weighting inputs validation successful")
            return True

        except Exception as e:
            self.logger.exception(f"Error validating weighting inputs: {e}")
            return False

    @handles_errors(fallback=None)
    async def _perform_performance_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform performance-based weighting."""
        try:
            # Define performance weighting methods
            weighting_methods = {
                "return_based_weighting": self._perform_return_based_weighting,
                "sharpe_based_weighting": self._perform_sharpe_based_weighting,
                "sortino_based_weighting": self._perform_sortino_based_weighting,
                "calmar_based_weighting": self._perform_calmar_based_weighting,
            }

            # Execute enabled weighting methods
            results = {}
            for method_name, method_func in weighting_methods.items():
                if self.performance_weighting_components.get(method_name, False):
                    results[method_name] = method_func(weighting_input)

            return results

        except Exception as e:
            self.logger.exception(f"Error performing performance weighting: {e}")
            return {}

    @handles_errors(fallback=None)
    async def _perform_risk_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform risk-based weighting."""
        try:
            # Define risk weighting methods
            weighting_methods = {
                "var_based_weighting": self._perform_var_based_weighting,
                "volatility_based_weighting": self._perform_volatility_based_weighting,
                "drawdown_based_weighting": self._perform_drawdown_based_weighting,
                "correlation_based_weighting": self._perform_correlation_based_weighting,
            }

            # Execute enabled weighting methods
            results = {}
            for method_name, method_func in weighting_methods.items():
                if self.risk_weighting_components.get(method_name, False):
                    results[method_name] = method_func(weighting_input)

            return results

        except Exception as e:
            self.logger.exception(f"Error performing risk weighting: {e}")
            return {}

    @handles_errors(fallback=None)
    async def _perform_adaptive_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform adaptive weighting."""
        try:
            results = {}

            # Market regime weighting
            if self.adaptive_weighting_components.get("market_regime_weighting", False):
                results["market_regime_weighting"] = self._perform_market_regime_weighting(weighting_input)

            # Regime detection
            if self.adaptive_weighting_components.get("regime_detection", False):
                results["regime_detection"] = self._perform_regime_detection(weighting_input)

            # Adaptive learning
            if self.adaptive_weighting_components.get("adaptive_learning", False):
                results["adaptive_learning"] = self._perform_adaptive_learning(weighting_input)

            # Dynamic adjustment
            if self.adaptive_weighting_components.get("dynamic_adjustment", False):
                results["dynamic_adjustment"] = self._perform_dynamic_adjustment(weighting_input)

            return results

        except Exception as e:
            self.logger.exception(f"Error performing adaptive weighting: {e}")
            return {}

    # Performance weighting methods

    def _perform_return_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform return based weighting."""
        try:
            # Simulate return based weighting
            return {
                "return_based_weighting_completed": True,
                "weighting_method": "return_based",
                "weights": [0.3, 0.25, 0.2, 0.15, 0.1],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing return based weighting: {e}")
            return {}

    def _perform_sharpe_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform Sharpe based weighting."""
        try:
            # Simulate Sharpe based weighting
            return {
                "sharpe_based_weighting_completed": True,
                "weighting_method": "sharpe_based",
                "weights": [0.35, 0.28, 0.22, 0.12, 0.03],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing Sharpe based weighting: {e}")
            return {}

    def _perform_sortino_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform Sortino based weighting."""
        try:
            # Simulate Sortino based weighting
            return {
                "sortino_based_weighting_completed": True,
                "weighting_method": "sortino_based",
                "weights": [0.32, 0.26, 0.21, 0.14, 0.07],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing Sortino based weighting: {e}")
            return {}

    def _perform_calmar_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform Calmar based weighting."""
        try:
            # Simulate Calmar based weighting
            return {
                "calmar_based_weighting_completed": True,
                "weighting_method": "calmar_based",
                "weights": [0.38, 0.30, 0.18, 0.10, 0.04],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing Calmar based weighting: {e}")
            return {}

    # Risk weighting methods

    def _perform_var_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform VaR based weighting."""
        try:
            # Simulate VaR based weighting
            return {
                "var_based_weighting_completed": True,
                "weighting_method": "var_based",
                "weights": [0.25, 0.25, 0.25, 0.15, 0.10],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing VaR based weighting: {e}")
            return {}

    def _perform_volatility_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volatility based weighting."""
        try:
            # Simulate volatility based weighting
            return {
                "volatility_based_weighting_completed": True,
                "weighting_method": "volatility_based",
                "weights": [0.20, 0.25, 0.30, 0.15, 0.10],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing volatility based weighting: {e}")
            return {}

    def _perform_drawdown_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform drawdown based weighting."""
        try:
            # Simulate drawdown based weighting
            return {
                "drawdown_based_weighting_completed": True,
                "weighting_method": "drawdown_based",
                "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing drawdown based weighting: {e}")
            return {}

    def _perform_correlation_based_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform correlation based weighting."""
        try:
            # Simulate correlation based weighting
            return {
                "correlation_based_weighting_completed": True,
                "weighting_method": "correlation_based",
                "weights": [0.35, 0.25, 0.20, 0.12, 0.08],
                "total_weight": 1.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing correlation based weighting: {e}")
            return {}

    # Adaptive weighting methods

    def _perform_market_regime_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform market regime weighting."""
        try:
            # Simulate market regime weighting
            return {
                "market_regime_weighting_completed": True,
                "weighting_method": "market_regime",
                "weights": [0.40, 0.30, 0.20, 0.08, 0.02],
                "regime": "bull_market",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing market regime weighting: {e}")
            return {}

    def _perform_regime_detection(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform regime detection."""
        try:
            # Simulate regime detection
            return {
                "regime_detection_completed": True,
                "detected_regime": "bull_market",
                "regime_probability": 0.75,
                "regime_confidence": 0.85,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing regime detection: {e}")
            return {}

    def _perform_regime_transition(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform regime transition."""
        try:
            # Simulate regime transition
            return {
                "regime_transition_completed": True,
                "transition_probability": 0.15,
                "transition_horizon": 5,
                "transition_confidence": 0.70,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing regime transition: {e}")
            return {}

    def _perform_regime_optimization(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform regime optimization."""
        try:
            # Simulate regime optimization
            return {
                "regime_optimization_completed": True,
                "optimization_method": "regime_based",
                "optimized_weights": [0.42, 0.28, 0.18, 0.08, 0.04],
                "optimization_score": 0.88,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing regime optimization: {e}")
            return {}

    def _perform_adaptive_learning(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform adaptive learning weighting."""
        try:
            # Simulate adaptive learning weighting
            return {
                "adaptive_learning_completed": True,
                "weighting_method": "adaptive_learning",
                "weights": [0.35, 0.25, 0.20, 0.12, 0.08],
                "learning_rate": 0.01,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing adaptive learning: {e}")
            return {}

    def _perform_dynamic_adjustment(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform dynamic adjustment weighting."""
        try:
            # Simulate dynamic adjustment weighting
            return {
                "dynamic_adjustment_completed": True,
                "weighting_method": "dynamic_adjustment",
                "weights": [0.30, 0.28, 0.22, 0.15, 0.05],
                "adjustment_factor": 0.85,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing dynamic adjustment: {e}")
            return {}

    # Momentum weighting methods

    def _perform_price_momentum(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform price momentum weighting."""
        try:
            # Simulate price momentum weighting
            return {
                "price_momentum_completed": True,
                "weighting_method": "price_momentum",
                "weights": [0.45, 0.25, 0.15, 0.10, 0.05],
                "momentum_score": 0.75,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing price momentum: {e}")
            return {}

    def _perform_volume_momentum(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volume momentum weighting."""
        try:
            # Simulate volume momentum weighting
            return {
                "volume_momentum_completed": True,
                "weighting_method": "volume_momentum",
                "weights": [0.40, 0.30, 0.20, 0.08, 0.02],
                "volume_score": 0.68,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing volume momentum: {e}")
            return {}

    def _perform_momentum_regime(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform momentum regime weighting."""
        try:
            # Simulate momentum regime weighting
            return {
                "momentum_regime_completed": True,
                "regime": "high_momentum",
                "regime_probability": 0.80,
                "weights": [0.50, 0.25, 0.15, 0.07, 0.03],
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing momentum regime: {e}")
            return {}

    def _perform_momentum_optimization(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform momentum optimization."""
        try:
            # Simulate momentum optimization
            return {
                "momentum_optimization_completed": True,
                "optimization_method": "momentum_based",
                "optimized_weights": [0.48, 0.26, 0.16, 0.07, 0.03],
                "optimization_score": 0.82,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing momentum optimization: {e}")
            return {}

    @handles_errors(fallback=None)
    async def _perform_momentum_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform momentum-based weighting."""
        try:
            results = {}

            # Price momentum weighting
            if self.momentum_weighting_components.get("price_momentum_weighting", False):
                results["price_momentum_weighting"] = self._perform_price_momentum_weighting(weighting_input)

            # Volume momentum weighting
            if self.momentum_weighting_components.get("volume_momentum_weighting", False):
                results["volume_momentum_weighting"] = self._perform_volume_momentum_weighting(weighting_input)

            # Momentum breakout weighting
            if self.momentum_weighting_components.get("momentum_breakout_weighting", False):
                results["momentum_breakout_weighting"] = self._perform_momentum_breakout_weighting(weighting_input)

            # Momentum reversal weighting
            if self.momentum_weighting_components.get("momentum_reversal_weighting", False):
                results["momentum_reversal_weighting"] = self._perform_momentum_reversal_weighting(weighting_input)

            return results

        except Exception as e:
            self.logger.exception(f"Error performing momentum weighting: {e}")
            return {}

    def _perform_price_momentum_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform price momentum weighting."""
        try:
            # Simulate price momentum weighting
            return {
                "price_momentum_weighting_completed": True,
                "weighting_method": "price_momentum",
                "weights": [0.45, 0.25, 0.15, 0.10, 0.05],
                "momentum_score": 0.75,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing price momentum weighting: {e}")
            return {}

    def _perform_volume_momentum_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volume momentum weighting."""
        try:
            # Simulate volume momentum weighting
            return {
                "volume_momentum_weighting_completed": True,
                "weighting_method": "volume_momentum",
                "weights": [0.40, 0.30, 0.20, 0.08, 0.02],
                "volume_score": 0.68,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing volume momentum weighting: {e}")
            return {}

    def _perform_momentum_breakout_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform momentum breakout weighting."""
        try:
            # Simulate momentum breakout weighting
            return {
                "momentum_breakout_weighting_completed": True,
                "weighting_method": "momentum_breakout",
                "weights": [0.50, 0.25, 0.15, 0.07, 0.03],
                "breakout_score": 0.82,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing momentum breakout weighting: {e}")
            return {}

    def _perform_momentum_reversal_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform momentum reversal weighting."""
        try:
            # Simulate momentum reversal weighting
            return {
                "momentum_reversal_weighting_completed": True,
                "weighting_method": "momentum_reversal",
                "weights": [0.35, 0.30, 0.20, 0.10, 0.05],
                "reversal_score": 0.65,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing momentum reversal weighting: {e}")
            return {}

    # Volatility weighting methods

    def _perform_historical_volatility_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform historical volatility weighting."""
        try:
            # Simulate historical volatility weighting
            return {
                "historical_volatility_weighting_completed": True,
                "weighting_method": "historical_volatility",
                "weights": [0.20, 0.25, 0.30, 0.15, 0.10],
                "volatility_score": 0.65,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(
                f"Error performing historical volatility weighting: {e}",
            )
            return {}

    def _perform_implied_volatility_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform implied volatility weighting."""
        try:
            # Simulate implied volatility weighting
            return {
                "implied_volatility_weighting_completed": True,
                "weighting_method": "implied_volatility",
                "weights": [0.18, 0.22, 0.35, 0.15, 0.10],
                "iv_score": 0.72,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing implied volatility weighting: {e}")
            return {}

    def _perform_volatility_regime_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volatility regime weighting."""
        try:
            # Simulate volatility regime weighting
            return {
                "volatility_regime_weighting_completed": True,
                "regime": "low_volatility",
                "regime_probability": 0.70,
                "weights": [0.25, 0.30, 0.25, 0.15, 0.05],
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing volatility regime weighting: {e}")
            return {}

    def _perform_volatility_optimization(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volatility optimization."""
        try:
            # Simulate volatility optimization
            return {
                "volatility_optimization_completed": True,
                "optimization_method": "volatility_based",
                "optimized_weights": [0.22, 0.28, 0.32, 0.13, 0.05],
                "optimization_score": 0.78,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing volatility optimization: {e}")
            return {}

    @handles_errors(fallback=None)
    async def _perform_volatility_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volatility-based weighting."""
        try:
            results = {}

            # Realized volatility weighting
            if self.volatility_weighting_components.get("realized_volatility_weighting", False):
                results["realized_volatility_weighting"] = self._perform_realized_volatility_weighting(weighting_input)

            # Implied volatility weighting
            if self.volatility_weighting_components.get("implied_volatility_weighting", False):
                results["implied_volatility_weighting"] = self._perform_implied_volatility_weighting(weighting_input)

            # Volatility regime weighting
            if self.volatility_weighting_components.get("volatility_regime_weighting", False):
                results["volatility_regime_weighting"] = self._perform_volatility_regime_weighting(weighting_input)

            # Volatility forecast weighting
            if self.volatility_weighting_components.get("volatility_forecast_weighting", False):
                results["volatility_forecast_weighting"] = self._perform_volatility_forecast_weighting(weighting_input)

            return results

        except Exception as e:
            self.logger.exception(f"Error performing volatility weighting: {e}")
            return {}

    def _perform_realized_volatility_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform realized volatility weighting."""
        try:
            # Simulate realized volatility weighting
            return {
                "realized_volatility_weighting_completed": True,
                "weighting_method": "realized_volatility",
                "weights": [0.20, 0.25, 0.30, 0.15, 0.10],
                "volatility_score": 0.65,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing realized volatility weighting: {e}")
            return {}

    def _perform_volatility_forecast_weighting(self, _weighting_input: dict[str, Any]) -> dict[str, Any]:
        """Perform volatility forecast weighting."""
        try:
            # Simulate volatility forecast weighting
            return {
                "volatility_forecast_weighting_completed": True,
                "weighting_method": "volatility_forecast",
                "weights": [0.22, 0.28, 0.32, 0.13, 0.05],
                "forecast_score": 0.78,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(f"Error performing volatility forecast weighting: {e}")
            return {}

    @handles_errors(fallback=None)
    async def _update_weighting_history(self) -> None:
        """Store weighting results."""
        try:
            # Add timestamp
            self.weighting_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.weighting_history.append(self.weighting_results.copy())

            # Limit history size
            if len(self.weighting_history) > self.max_weighting_history:
                self.weighting_history.pop(0)

            self.logger.info("Weighting results stored successfully")

        except Exception as e:
            self.logger.exception(f"Error storing weighting results: {e}")

    @handles_errors(fallback=None)
    def get_weighting_results(self, weighting_type: str | None = None) -> dict[str, Any]:
        """
        Get weighting results.

        Args:
            weighting_type: Optional weighting type filter

        Returns:
            dict[str, Any]: Weighting results
        """
        try:
            if weighting_type:
                return self.weighting_results.get(weighting_type, {})
            return self.weighting_results.copy()

        except Exception as e:
            self.logger.exception(f"Error getting weighting results: {e}")
            return {}

    @handles_errors(fallback=None)
    def get_weighting_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get weighting history.

        Args:
            limit: Optional limit on number of records

        Returns:
            list[dict[str, Any]]: Weighting history
        """
        try:
            history = self.weighting_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.exception(f"Error getting weighting history: {e}")
            return []

    def get_weighting_status(self) -> dict[str, Any]:
        """
        Get weighting status information.

        Returns:
            dict[str, Any]: Weighting status
        """
        return {
            "is_weighting": self.is_weighting,
            "weighting_interval": self.weighting_interval,
            "max_weighting_history": self.max_weighting_history,
            "enable_performance_weighting": self.enable_performance_weighting,
            "enable_risk_weighting": self.enable_risk_weighting,
            "enable_adaptive_weighting": self.enable_adaptive_weighting,
            "enable_online_learning": self.enable_online_learning,
            "enable_regime_awareness": self.enable_regime_awareness,
            "enable_uncertainty_weighting": self.enable_uncertainty_weighting,
            "enable_momentum_weighting": self.weighter_config.get("enable_momentum_weighting", True),
            "enable_volatility_weighting": self.weighter_config.get("enable_volatility_weighting", True),
            "weighting_history_count": len(self.weighting_history),
            "model_weights": self.model_weights.copy(),
            "model_performances": {k: len(v) for k, v in self.model_performances.items()},
        }

    # ============================================================================
    # ENHANCED ENSEMBLE WEIGHTING METHODS
    # ============================================================================

    @handles_errors(fallback=None)
    async def update_model_weights_online(
        self,
        model_predictions: dict[str, float],
        actual_outcomes: dict[str, float],
        timestamp: datetime = None,
    ) -> None:
        """Update model weights using online learning."""
        try:
            if not self.enable_online_learning:
                return

            for model_name, prediction in model_predictions.items():
                if model_name not in actual_outcomes:
                    continue

                actual_outcome = actual_outcomes[model_name]

                # Calculate prediction error
                error = abs(prediction - actual_outcome)

                # Initialize weight if not exists
                if model_name not in self.model_weights:
                    self.model_weights[model_name] = 1.0

                # Initialize performance history if not exists
                if model_name not in self.model_performances:
                    self.model_performances[model_name] = deque(maxlen=self.performance_window)

                # Store performance data
                performance_data = {
                    "prediction": prediction,
                    "actual": actual_outcome,
                    "error": error,
                    "timestamp": timestamp or datetime.now(),
                }
                self.model_performances[model_name].append(performance_data)  # O(1) with automatic size limit

                # Update weight using gradient descent
                # Inverse relationship: higher error = lower weight
                weight_gradient = -error * self.learning_rate
                self.model_weights[model_name] += weight_gradient

                # Ensure weights are positive
                self.model_weights[model_name] = max(0.01, self.model_weights[model_name])

            # Normalize weights to sum to 1
            await self._normalize_weights()

            self.logger.info(f"Updated model weights: {self.model_weights}")

        except Exception as e:
            self.logger.exception(f"Error updating model weights online: {e}")

    @handles_errors(fallback=None)
    async def get_regime_aware_weights(self, current_regime: str, model_names: list[str]) -> dict[str, float]:
        """Get regime-specific ensemble weights."""
        try:
            if not self.enable_regime_awareness:
                # Return equal weights if regime awareness is disabled
                return {model: 1.0 / len(model_names) for model in model_names}

            # Get regime-specific base weights
            base_weights = self._get_regime_base_weights(current_regime)

            # Initialize regime performance tracking
            if current_regime not in self.regime_performances:
                self.regime_performances[current_regime] = {}

            # Calculate regime-specific weights
            regime_weights_result = {}
            for model_name in model_names:
                # Get base weight for this model in this regime
                base_weight = base_weights.get(model_name, 0.2)

                # Adjust based on recent performance in this regime
                recent_performance = self._get_recent_regime_performance(model_name, current_regime)
                performance_multiplier = 0.5 + recent_performance  # 0.5-1.5 range

                regime_weights_result[model_name] = base_weight * performance_multiplier

            # Normalize weights
            total_weight = sum(regime_weights_result.values())
            if total_weight > 0:
                regime_weights_result = {k: v / total_weight for k, v in regime_weights_result.items()}

            return regime_weights_result

        except Exception as e:
            self.logger.exception(f"Error calculating regime-aware weights: {e}")
            return {model: 1.0 / len(model_names) for model in model_names}

    @handles_errors(fallback=None)
    async def get_uncertainty_aware_weights(
        self, model_predictions: dict[str, float], model_uncertainties: dict[str, float],
    ) -> dict[str, float]:
        """Get uncertainty-aware ensemble weights."""
        try:
            if not self.enable_uncertainty_weighting:
                # Return equal weights if uncertainty weighting is disabled
                return {model: 1.0 / len(model_predictions) for model in model_predictions}

            weights = {}
            total_inverse_uncertainty = 0.0

            for model_name, uncertainty in model_uncertainties.items():
                if model_name not in model_predictions:
                    continue

                # Models with lower uncertainty get higher weights
                # Add small epsilon to avoid division by zero
                inverse_uncertainty = 1.0 / (uncertainty + 1e-6)
                weights[model_name] = inverse_uncertainty
                total_inverse_uncertainty += inverse_uncertainty

            # Normalize weights
            if total_inverse_uncertainty > 0:
                weights = {k: v / total_inverse_uncertainty for k, v in weights.items()}

            return weights

        except Exception as e:
            self.logger.exception(f"Error calculating uncertainty-aware weights: {e}")
            return {model: 1.0 / len(model_predictions) for model in model_predictions}

    @handles_errors(fallback=None)
    async def calculate_enhanced_ensemble_weights(
        self,
        model_predictions: dict[str, float],
        model_uncertainties: dict[str, float],
        current_regime: str = None,
    ) -> dict[str, float]:
        """Calculate enhanced ensemble weights combining multiple factors."""
        try:
            model_names = list(model_predictions.keys())

            # Get different types of weights
            online_weights = self.model_weights.copy()
            regime_weights = await self.get_regime_aware_weights(current_regime, model_names)
            uncertainty_weights = await self.get_uncertainty_aware_weights(model_predictions, model_uncertainties)

            # Combine weights with configurable importance
            combined_weights = {}
            for model_name in model_names:
                online_weight = online_weights.get(model_name, 1.0 / len(model_names))
                regime_weight = regime_weights.get(model_name, 1.0 / len(model_names))
                uncertainty_weight = uncertainty_weights.get(model_name, 1.0 / len(model_names))

                # Weight combination (can be made configurable)
                combined_weight = (
                    0.4 * online_weight  # 40% online learning
                    + 0.4 * regime_weight  # 40% regime awareness
                    + 0.2 * uncertainty_weight  # 20% uncertainty
                )

                combined_weights[model_name] = combined_weight

            # Normalize final weights
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {k: v / total_weight for k, v in combined_weights.items()}

            self.logger.info(f"Enhanced ensemble weights: {combined_weights}")
            return combined_weights

        except Exception as e:
            self.logger.exception(f"Error calculating enhanced ensemble weights: {e}")
            return {model: 1.0 / len(model_predictions) for model in model_predictions}

    def _get_regime_base_weights(self, regime: str) -> dict[str, float]:
        """Get base weights for models in a specific regime."""
        regime_weights = {
            "BULL": {
                "tcn": 0.4,
                "transformer": 0.3,
                "lstm": 0.3,
                "gru": 0.2,
                "tabnet": 0.3,
            },
            "BEAR": {
                "tcn": 0.3,
                "transformer": 0.4,
                "lstm": 0.3,
                "gru": 0.3,
                "tabnet": 0.2,
            },
            "SIDEWAYS": {
                "tcn": 0.3,
                "transformer": 0.3,
                "lstm": 0.4,
                "gru": 0.3,
                "tabnet": 0.3,
            },
            "SR": {
                "tcn": 0.5,
                "transformer": 0.3,
                "lstm": 0.2,
                "gru": 0.2,
                "tabnet": 0.4,
            },
            "CANDLE": {
                "tcn": 0.3,
                "transformer": 0.5,
                "lstm": 0.3,
                "gru": 0.3,
                "tabnet": 0.2,
            },
        }
        return regime_weights.get(regime, {})

    def _get_recent_regime_performance(self, model_name: str, regime: str) -> float:
        """Get recent performance of a model in a specific regime."""
        try:
            if model_name not in self.model_performances:
                return 0.5  # Default performance

            # Get recent performance data
            recent_data = self.model_performances[model_name][-20:]  # Last 20 predictions
            if not recent_data:
                return 0.5

            # Calculate average accuracy (assuming lower error = better performance)
            total_error = sum(d["error"] for d in recent_data)
            avg_error = total_error / len(recent_data)

            # Convert error to performance score (0-1, higher is better)
            return max(0.0, 1.0 - avg_error)


        except Exception as e:
            self.logger.exception(f"Error getting recent regime performance: {e}")
            return 0.5

    @handles_errors(fallback=None)
    async def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1."""
        try:
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}
            else:
                # If all weights are zero = set equal weights
                model_count = len(self.model_weights)
                if model_count > 0:
                    self.model_weights = dict.fromkeys(self.model_weights.keys(), 1.0 / model_count)

        except Exception as e:
            self.logger.exception(f"Error normalizing weights: {e}")

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the dynamic weighter."""
        self.logger.info("🛑 Stopping Dynamic Weighter...")

        try:
            # Stop weighting
            self.is_weighting = False

            # Clear results
            self.weighting_results.clear()

            # Clear history
            self.weighting_history.clear()

            self.logger.info("✅ Dynamic Weighter stopped successfully")

        except Exception as e:
            self.logger.exception(f"Error stopping dynamic weighter: {e}")

# Global dynamic weighter instance
dynamic_weighter: DynamicWeighter | None = None

@handles_errors(fallback=None)
async def setup_dynamic_weighter(
    config: dict[str, Any] | None = None,
) -> DynamicWeighter | None:
    """
    Setup global dynamic weighter.

    Args:
        config: Optional configuration dictionary

    Returns:
        DynamicWeighter | None: Global dynamic weighter instance
    """
    try:
        global dynamic_weighter

        if config is None:
            config = {
                "dynamic_weighter": {
                    "weighting_interval": 3600,
                    "max_weighting_history": 100,
                    "enable_performance_weighting": True,
                    "enable_risk_weighting": True,
                    "enable_adaptive_weighting": True,
                    "enable_momentum_weighting": True,
                    "enable_volatility_weighting": True,
                },
            }

        # Create dynamic weighter
        dynamic_weighter = DynamicWeighter(config)

        # Initialize dynamic weighter
        success = await dynamic_weighter.initialize()
        if success:
            return dynamic_weighter
        return None

    except Exception as e:
        print(f"Error setting up dynamic weighter: {e}")
        return None
