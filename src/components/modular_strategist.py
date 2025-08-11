# src/components/modular_strategist.py

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    initialization_error,
    invalid,
    missing,
)


class ModularStrategist:
    """
    Enhanced modular strategist with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize modular strategist with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ModularStrategist")

        # Strategy state
        self.is_strategizing: bool = False
        self.strategy_results: dict[str, Any] = {}
        self.strategy_history: list[dict[str, Any]] = []

        # Configuration
        self.strategist_config: dict[str, Any] = self.config.get(
            "modular_strategist",
            {},
        )
        self.strategy_interval: int = self.strategist_config.get(
            "strategy_interval",
            60,
        )
        self.max_strategy_history: int = self.strategist_config.get(
            "max_strategy_history",
            100,
        )
        self.enable_position_sizing: bool = self.strategist_config.get(
            "enable_position_sizing",
            True,
        )
        self.enable_risk_management: bool = self.strategist_config.get(
            "enable_risk_management",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular strategist configuration"),
            AttributeError: (False, "Missing required strategist parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular strategist initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize modular strategist with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Modular Strategist...")

        # Load strategist configuration
        await self._load_strategist_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.print(invalid("Invalid configuration for modular strategist"))
            return False

        # Initialize strategy modules
        await self._initialize_strategy_modules()

        self.logger.info(
            "✅ Modular Strategist initialization completed successfully",
        )
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategist configuration loading",
    )
    async def _load_strategist_configuration(self) -> None:
        """Load strategist configuration."""
        # Set default strategist parameters
        self.strategist_config.setdefault("strategy_interval", 60)
        self.strategist_config.setdefault("max_strategy_history", 100)
        self.strategist_config.setdefault("enable_position_sizing", True)
        self.strategist_config.setdefault("enable_risk_management", True)
        self.strategist_config.setdefault("enable_portfolio_optimization", False)
        self.strategist_config.setdefault("enable_dynamic_rebalancing", True)

        # Update configuration
        self.strategy_interval = self.strategist_config["strategy_interval"]
        self.max_strategy_history = self.strategist_config["max_strategy_history"]
        self.enable_position_sizing = self.strategist_config["enable_position_sizing"]
        self.enable_risk_management = self.strategist_config["enable_risk_management"]

        self.logger.info("Strategist configuration loaded successfully")

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
        # Validate strategy interval
        if self.strategy_interval <= 0:
            self.print(invalid("Invalid strategy interval"))
            return False

        # Validate max strategy history
        if self.max_strategy_history <= 0:
            self.print(invalid("Invalid max strategy history"))
            return False

        # Validate that at least one strategy type is enabled
        if not any(
            [
                self.enable_position_sizing,
                self.enable_risk_management,
                self.strategist_config.get("enable_portfolio_optimization", False),
                self.strategist_config.get("enable_dynamic_rebalancing", True),
            ],
        ):
            self.print(error("At least one strategy type must be enabled"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy modules initialization",
    )
    async def _initialize_strategy_modules(self) -> None:
        """Initialize strategy modules."""
        try:
            # Initialize position sizing module
            if self.enable_position_sizing:
                await self._initialize_position_sizing()

            # Initialize risk management module
            if self.enable_risk_management:
                await self._initialize_risk_management()

            # Initialize portfolio optimization module
            if self.strategist_config.get("enable_portfolio_optimization", False):
                await self._initialize_portfolio_optimization()

            # Initialize dynamic rebalancing module
            if self.strategist_config.get("enable_dynamic_rebalancing", True):
                await self._initialize_dynamic_rebalancing()

            self.logger.info("Strategy modules initialized successfully")

        except Exception:
            self.print(initialization_error("Error initializing strategy modules: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position sizing initialization",
    )
    async def _initialize_position_sizing(self) -> None:
        """Initialize position sizing module."""
        try:
            # Initialize position sizing strategies
            self.position_sizing_strategies = {
                "kelly_criterion": True,
                "fixed_fraction": True,
                "volatility_targeting": True,
                "risk_parity": True,
            }

            self.logger.info("Position sizing module initialized")

        except Exception:
            self.print(initialization_error("Error initializing position sizing: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk management initialization",
    )
    async def _initialize_risk_management(self) -> None:
        """Initialize risk management module."""
        try:
            # Initialize risk management strategies
            self.risk_management_strategies = {
                "stop_loss": True,
                "take_profit": True,
                "trailing_stop": True,
                "position_limits": True,
            }

            self.logger.info("Risk management module initialized")

        except Exception:
            self.print(initialization_error("Error initializing risk management: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="portfolio optimization initialization",
    )
    async def _initialize_portfolio_optimization(self) -> None:
        """Initialize portfolio optimization module."""
        try:
            # Initialize portfolio optimization strategies
            self.portfolio_optimization_strategies = {
                "mean_variance": True,
                "black_litterman": True,
                "risk_parity": True,
                "maximum_sharpe": True,
            }

            self.logger.info("Portfolio optimization module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing portfolio optimization: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dynamic rebalancing initialization",
    )
    async def _initialize_dynamic_rebalancing(self) -> None:
        """Initialize dynamic rebalancing module."""
        try:
            # Initialize dynamic rebalancing strategies
            self.dynamic_rebalancing_strategies = {
                "threshold_rebalancing": True,
                "calendar_rebalancing": True,
                "drift_rebalancing": True,
                "volatility_rebalancing": True,
            }

            self.logger.info("Dynamic rebalancing module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing dynamic rebalancing: {e}"),
            )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid strategy parameters"),
            AttributeError: (False, "Missing strategy components"),
            KeyError: (False, "Missing required strategy data"),
        },
        default_return=False,
        context="strategy execution",
    )
    async def execute_strategy(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> bool:
        """
        Execute trading strategy.

        Args:
            market_data: Market data dictionary
            analysis_data: Analysis data dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_strategy_inputs(market_data, analysis_data):
                return False

            self.is_strategizing = True
            self.logger.info("🔄 Starting strategy execution...")

            # Perform position sizing
            if self.enable_position_sizing:
                position_results = await self._perform_position_sizing(
                    market_data,
                    analysis_data,
                )
                self.strategy_results["position_sizing"] = position_results

            # Perform risk management
            if self.enable_risk_management:
                risk_results = await self._perform_risk_management(
                    market_data,
                    analysis_data,
                )
                self.strategy_results["risk_management"] = risk_results

            # Perform portfolio optimization
            if self.strategist_config.get("enable_portfolio_optimization", False):
                portfolio_results = await self._perform_portfolio_optimization(
                    market_data,
                    analysis_data,
                )
                self.strategy_results["portfolio_optimization"] = portfolio_results

            # Perform dynamic rebalancing
            if self.strategist_config.get("enable_dynamic_rebalancing", True):
                rebalancing_results = await self._perform_dynamic_rebalancing(
                    market_data,
                    analysis_data,
                )
                self.strategy_results["dynamic_rebalancing"] = rebalancing_results

            # Store strategy results
            await self._store_strategy_results()

            self.is_strategizing = False
            self.logger.info("✅ Strategy execution completed successfully")
            return True

        except Exception:
            self.print(error("Error executing strategy: {e}"))
            self.is_strategizing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="strategy inputs validation",
    )
    def _validate_strategy_inputs(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> bool:
        """
        Validate strategy inputs.

        Args:
            market_data: Market data dictionary
            analysis_data: Analysis data dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required market data fields
            required_market_fields = ["symbol", "price", "volume", "timestamp"]
            for field in required_market_fields:
                if field not in market_data:
                    self.print(missing("Missing required market data field: {field}"))
                    return False

            # Check required analysis data fields
            required_analysis_fields = ["signal", "confidence"]
            for field in required_analysis_fields:
                if field not in analysis_data:
                    self.print(missing("Missing required analysis data field: {field}"))
                    return False

            # Validate data types
            if not isinstance(market_data["price"], int | float):
                self.print(invalid("Invalid price data type"))
                return False

            if not isinstance(analysis_data["confidence"], int | float):
                self.print(invalid("Invalid confidence data type"))
                return False

            return True

        except Exception:
            self.print(error("Error validating strategy inputs: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position sizing",
    )
    async def _perform_position_sizing(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform position sizing.

        Args:
            market_data: Market data dictionary
            analysis_data: Analysis data dictionary

        Returns:
            Dict[str, Any]: Position sizing results
        """
        try:
            results = {}

            # Calculate Kelly Criterion
            if self.position_sizing_strategies.get("kelly_criterion", False):
                results["kelly_criterion"] = self._calculate_kelly_criterion(
                    market_data,
                    analysis_data,
                )

            # Calculate Fixed Fraction
            if self.position_sizing_strategies.get("fixed_fraction", False):
                results["fixed_fraction"] = self._calculate_fixed_fraction(
                    market_data,
                    analysis_data,
                )

            # Calculate Volatility Targeting
            if self.position_sizing_strategies.get("volatility_targeting", False):
                results["volatility_targeting"] = self._calculate_volatility_targeting(
                    market_data,
                    analysis_data,
                )

            # Calculate Risk Parity
            if self.position_sizing_strategies.get("risk_parity", False):
                results["risk_parity"] = self._calculate_risk_parity(
                    market_data,
                    analysis_data,
                )

            self.logger.info("Position sizing completed")
            return results

        except Exception:
            self.print(error("Error performing position sizing: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk management",
    )
    async def _perform_risk_management(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform risk management.

        Args:
            market_data: Market data dictionary
            analysis_data: Analysis data dictionary

        Returns:
            Dict[str, Any]: Risk management results
        """
        try:
            results = {}

            # Calculate Stop Loss
            if self.risk_management_strategies.get("stop_loss", False):
                results["stop_loss"] = self._calculate_stop_loss(
                    market_data,
                    analysis_data,
                )

            # Calculate Take Profit
            if self.risk_management_strategies.get("take_profit", False):
                results["take_profit"] = self._calculate_take_profit(
                    market_data,
                    analysis_data,
                )

            # Calculate Trailing Stop
            if self.risk_management_strategies.get("trailing_stop", False):
                results["trailing_stop"] = self._calculate_trailing_stop(
                    market_data,
                    analysis_data,
                )

            # Calculate Position Limits
            if self.risk_management_strategies.get("position_limits", False):
                results["position_limits"] = self._calculate_position_limits(
                    market_data,
                    analysis_data,
                )

            self.logger.info("Risk management completed")
            return results

        except Exception:
            self.print(error("Error performing risk management: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="portfolio optimization",
    )
    async def _perform_portfolio_optimization(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform portfolio optimization.

        Args:
            market_data: Market data dictionary
            analysis_data: Analysis data dictionary

        Returns:
            Dict[str, Any]: Portfolio optimization results
        """
        try:
            results = {}

            # Calculate Mean Variance
            if self.portfolio_optimization_strategies.get("mean_variance", False):
                results["mean_variance"] = self._calculate_mean_variance(
                    market_data,
                    analysis_data,
                )

            # Calculate Black Litterman
            if self.portfolio_optimization_strategies.get("black_litterman", False):
                results["black_litterman"] = self._calculate_black_litterman(
                    market_data,
                    analysis_data,
                )

            # Calculate Risk Parity
            if self.portfolio_optimization_strategies.get("risk_parity", False):
                results["risk_parity"] = self._calculate_portfolio_risk_parity(
                    market_data,
                    analysis_data,
                )

            # Calculate Maximum Sharpe
            if self.portfolio_optimization_strategies.get("maximum_sharpe", False):
                results["maximum_sharpe"] = self._calculate_maximum_sharpe(
                    market_data,
                    analysis_data,
                )

            self.logger.info("Portfolio optimization completed")
            return results

        except Exception:
            self.print(error("Error performing portfolio optimization: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dynamic rebalancing",
    )
    async def _perform_dynamic_rebalancing(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform dynamic rebalancing.

        Args:
            market_data: Market data dictionary
            analysis_data: Analysis data dictionary

        Returns:
            Dict[str, Any]: Dynamic rebalancing results
        """
        try:
            results = {}

            # Calculate Threshold Rebalancing
            if self.dynamic_rebalancing_strategies.get("threshold_rebalancing", False):
                results["threshold_rebalancing"] = (
                    self._calculate_threshold_rebalancing(market_data, analysis_data)
                )

            # Calculate Calendar Rebalancing
            if self.dynamic_rebalancing_strategies.get("calendar_rebalancing", False):
                results["calendar_rebalancing"] = self._calculate_calendar_rebalancing(
                    market_data,
                    analysis_data,
                )

            # Calculate Drift Rebalancing
            if self.dynamic_rebalancing_strategies.get("drift_rebalancing", False):
                results["drift_rebalancing"] = self._calculate_drift_rebalancing(
                    market_data,
                    analysis_data,
                )

            # Calculate Volatility Rebalancing
            if self.dynamic_rebalancing_strategies.get("volatility_rebalancing", False):
                results["volatility_rebalancing"] = (
                    self._calculate_volatility_rebalancing(market_data, analysis_data)
                )

            self.logger.info("Dynamic rebalancing completed")
            return results

        except Exception:
            self.print(error("Error performing dynamic rebalancing: {e}"))
            return {}

    # Position sizing calculation methods
    def _calculate_kelly_criterion(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Kelly Criterion position size."""
        try:
            # Simulate Kelly Criterion calculation
            win_rate = analysis_data.get("confidence", 0.5)
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        except Exception:
            self.print(error("Error calculating Kelly Criterion: {e}"))
            return 0.0

    def _calculate_fixed_fraction(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Fixed Fraction position size."""
        try:
            # Simulate Fixed Fraction calculation
            confidence = analysis_data.get("confidence", 0.5)
            base_fraction = 0.1  # 10% base position

            return base_fraction * confidence
        except Exception:
            self.print(error("Error calculating Fixed Fraction: {e}"))
            return 0.0

    def _calculate_volatility_targeting(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Volatility Targeting position size."""
        try:
            # Simulate Volatility Targeting calculation
            volatility = 0.02  # 2% volatility
            target_volatility = 0.01  # 1% target volatility

            return target_volatility / volatility
        except Exception:
            self.print(error("Error calculating Volatility Targeting: {e}"))
            return 0.0

    def _calculate_risk_parity(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Risk Parity position size."""
        try:
            # Simulate Risk Parity calculation
            return 0.5  # Equal risk contribution

        except Exception:
            self.print(error("Error calculating Risk Parity: {e}"))
            return 0.0

    # Risk management calculation methods
    def _calculate_stop_loss(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Stop Loss level."""
        try:
            # Simulate Stop Loss calculation
            current_price = market_data.get("price", 0)
            stop_loss_pct = 0.02  # 2% stop loss

            return current_price * (1 - stop_loss_pct)
        except Exception:
            self.print(error("Error calculating Stop Loss: {e}"))
            return 0.0

    def _calculate_take_profit(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Take Profit level."""
        try:
            # Simulate Take Profit calculation
            current_price = market_data.get("price", 0)
            take_profit_pct = 0.04  # 4% take profit

            return current_price * (1 + take_profit_pct)
        except Exception:
            self.print(error("Error calculating Take Profit: {e}"))
            return 0.0

    def _calculate_trailing_stop(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> float:
        """Calculate Trailing Stop level."""
        try:
            # Simulate Trailing Stop calculation
            current_price = market_data.get("price", 0)
            trailing_pct = 0.015  # 1.5% trailing stop

            return current_price * (1 - trailing_pct)
        except Exception:
            self.print(error("Error calculating Trailing Stop: {e}"))
            return 0.0

    def _calculate_position_limits(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate Position Limits."""
        try:
            # Simulate Position Limits calculation
            return {
                "max_position_size": 0.25,  # 25% max position
                "max_leverage": 3.0,  # 3x max leverage
                "max_drawdown": 0.1,  # 10% max drawdown
            }
        except Exception:
            self.print(error("Error calculating Position Limits: {e}"))
            return {"max_position_size": 0.0, "max_leverage": 0.0, "max_drawdown": 0.0}

    # Portfolio optimization calculation methods
    def _calculate_mean_variance(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate Mean Variance optimization."""
        try:
            # Simulate Mean Variance calculation
            return {
                "optimal_weight": 0.6,
                "expected_return": 0.08,
                "volatility": 0.15,
                "sharpe_ratio": 0.53,
            }
        except Exception:
            self.print(error("Error calculating Mean Variance: {e}"))
            return {
                "optimal_weight": 0.0,
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
            }

    def _calculate_black_litterman(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate Black Litterman optimization."""
        try:
            # Simulate Black Litterman calculation
            return {
                "optimal_weight": 0.55,
                "expected_return": 0.07,
                "volatility": 0.14,
                "confidence": 0.8,
            }
        except Exception:
            self.print(error("Error calculating Black Litterman: {e}"))
            return {
                "optimal_weight": 0.0,
                "expected_return": 0.0,
                "volatility": 0.0,
                "confidence": 0.0,
            }

    def _calculate_portfolio_risk_parity(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate Portfolio Risk Parity."""
        try:
            # Simulate Portfolio Risk Parity calculation
            return {
                "risk_contribution": 0.5,
                "volatility": 0.12,
                "diversification_ratio": 1.2,
            }
        except Exception:
            self.print(error("Error calculating Portfolio Risk Parity: {e}"))
            return {
                "risk_contribution": 0.0,
                "volatility": 0.0,
                "diversification_ratio": 0.0,
            }

    def _calculate_maximum_sharpe(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate Maximum Sharpe optimization."""
        try:
            # Simulate Maximum Sharpe calculation
            return {
                "optimal_weight": 0.65,
                "expected_return": 0.09,
                "volatility": 0.16,
                "sharpe_ratio": 0.56,
            }
        except Exception:
            self.print(error("Error calculating Maximum Sharpe: {e}"))
            return {
                "optimal_weight": 0.0,
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
            }

    # Dynamic rebalancing calculation methods
    def _calculate_threshold_rebalancing(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> bool:
        """Calculate Threshold Rebalancing trigger."""
        try:
            # Simulate Threshold Rebalancing calculation
            drift = np.random.random() * 0.1  # Random drift
            threshold = 0.05  # 5% threshold

            return drift > threshold
        except Exception:
            self.print(error("Error calculating Threshold Rebalancing: {e}"))
            return False

    def _calculate_calendar_rebalancing(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> bool:
        """Calculate Calendar Rebalancing trigger."""
        try:
            # Simulate Calendar Rebalancing calculation
            current_time = datetime.now()
            last_rebalance = datetime.now() - timedelta(days=30)
            rebalance_interval = timedelta(days=7)  # Weekly rebalancing

            return (current_time - last_rebalance) > rebalance_interval
        except Exception:
            self.print(error("Error calculating Calendar Rebalancing: {e}"))
            return False

    def _calculate_drift_rebalancing(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> bool:
        """Calculate Drift Rebalancing trigger."""
        try:
            # Simulate Drift Rebalancing calculation
            drift = np.random.random() * 0.08  # Random drift
            max_drift = 0.03  # 3% max drift

            return drift > max_drift
        except Exception:
            self.print(error("Error calculating Drift Rebalancing: {e}"))
            return False

    def _calculate_volatility_rebalancing(
        self,
        market_data: dict[str, Any],
        analysis_data: dict[str, Any],
    ) -> bool:
        """Calculate Volatility Rebalancing trigger."""
        try:
            # Simulate Volatility Rebalancing calculation
            current_volatility = np.random.random() * 0.05  # Random volatility
            target_volatility = 0.02  # 2% target volatility
            threshold = 0.01  # 1% threshold

            return abs(current_volatility - target_volatility) > threshold
        except Exception:
            self.print(error("Error calculating Volatility Rebalancing: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy results storage",
    )
    async def _store_strategy_results(self) -> None:
        """Store strategy results."""
        try:
            # Add timestamp
            self.strategy_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.strategy_history.append(self.strategy_results.copy())

            # Limit history size
            if len(self.strategy_history) > self.max_strategy_history:
                self.strategy_history.pop(0)

            self.logger.info("Strategy results stored successfully")

        except Exception:
            self.print(error("Error storing strategy results: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy results getting",
    )
    def get_strategy_results(
        self,
        strategy_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get strategy results.

        Args:
            strategy_type: Optional strategy type filter

        Returns:
            Dict[str, Any]: Strategy results
        """
        try:
            if strategy_type:
                return self.strategy_results.get(strategy_type, {})
            return self.strategy_results.copy()

        except Exception:
            self.print(error("Error getting strategy results: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy history getting",
    )
    def get_strategy_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get strategy history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Strategy history
        """
        try:
            history = self.strategy_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception:
            self.print(error("Error getting strategy history: {e}"))
            return []

    def get_strategist_status(self) -> dict[str, Any]:
        """
        Get strategist status information.

        Returns:
            Dict[str, Any]: Strategist status
        """
        return {
            "is_strategizing": self.is_strategizing,
            "strategy_interval": self.strategy_interval,
            "max_strategy_history": self.max_strategy_history,
            "enable_position_sizing": self.enable_position_sizing,
            "enable_risk_management": self.enable_risk_management,
            "enable_portfolio_optimization": self.strategist_config.get(
                "enable_portfolio_optimization",
                False,
            ),
            "enable_dynamic_rebalancing": self.strategist_config.get(
                "enable_dynamic_rebalancing",
                True,
            ),
            "strategy_history_count": len(self.strategy_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="modular strategist cleanup",
    )
    async def stop(self) -> None:
        """Stop the modular strategist."""
        self.logger.info("🛑 Stopping Modular Strategist...")

        try:
            # Stop strategizing
            self.is_strategizing = False

            # Clear results
            self.strategy_results.clear()

            # Clear history
            self.strategy_history.clear()

            self.logger.info("✅ Modular Strategist stopped successfully")

        except Exception:
            self.print(error("Error stopping modular strategist: {e}"))


# Global modular strategist instance
modular_strategist: ModularStrategist | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="modular strategist setup",
)
async def setup_modular_strategist(
    config: dict[str, Any] | None = None,
) -> ModularStrategist | None:
    """
    Setup global modular strategist.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[ModularStrategist]: Global modular strategist instance
    """
    try:
        global modular_strategist

        if config is None:
            config = {
                "modular_strategist": {
                    "strategy_interval": 60,
                    "max_strategy_history": 100,
                    "enable_position_sizing": True,
                    "enable_risk_management": True,
                    "enable_portfolio_optimization": False,
                    "enable_dynamic_rebalancing": True,
                },
            }

        # Create modular strategist
        modular_strategist = ModularStrategist(config)

        # Initialize modular strategist
        success = await modular_strategist.initialize()
        if success:
            return modular_strategist
        return None

    except Exception as e:
        print(f"Error setting up modular strategist: {e}")
        return None
