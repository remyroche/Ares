from src.utils.tprint import tprint


from datetime import datetime, timedelta
from typing import Any
import numpy as np
import asyncio

from logging import error

from ..utils.logger import system_logger
from ..utils.warning_symbols import initialization_error, invalid, missing
from ..core.decorators import handles_errors
from ..interfaces.base_interfaces import IStrategist, AnalysisResult, StrategyResult
import time

# src/components/modular_strategist.py

class ModularStrategist(IStrategist):
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

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid modular strategist configuration"),
            AttributeError: (False, "Missing required strategist parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return = False,
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
            self.logger.error(invalid("Invalid configuration for modular strategist"))
            return False

        # Initialize strategy modules
        await self._initialize_strategy_modules()

        self.logger.info(
            "✅ Modular Strategist initialization completed successfully",
        )
        return True

    @handles_errors(fallback = None)
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

    @handles_errors(fallback = False)
    def _validate_configuration(self) -> bool:
        """
        Validate strategist configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate strategy interval
        if self.strategy_interval <= 0:
            self.logger.error(invalid("Invalid strategy interval"))
            return False

        # Validate max strategy history
        if self.max_strategy_history <= 0:
            self.logger.error(invalid("Invalid max strategy history"))
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
            self.logger.error(error("At least one strategy type must be enabled"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing strategy modules: {e}")
            )

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing position sizing: {e}")
            )

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing risk management: {e}")
            )

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing portfolio optimization: {e}"),
            )

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing dynamic rebalancing: {e}"),
            )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid strategy parameters"),
            AttributeError: (False, "Missing strategy components"),
            KeyError: (False, "Missing required strategy data"),
        },
        default_return = False,
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

        except Exception as e:
            self.logger.exception(error(f"Error executing strategy: {e}"))
            self.is_strategizing = False
            return False

    @handles_errors(fallback = False)
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
                    self.logger.error(
                        missing(f"Missing required market data field: {field}")
                    )
                    return False

            # Check required analysis data fields
            required_analysis_fields = ["signal", "confidence"]
            for field in required_analysis_fields:
                if field not in analysis_data:
                    self.logger.error(
                        missing(f"Missing required analysis data field: {field}")
                    )
                    return False

            # Validate data types
            if not isinstance(market_data["price"], int | float):
                self.logger.error(invalid("Invalid price data type"))
                return False

            if not isinstance(analysis_data["confidence"], int | float):
                self.logger.error(invalid("Invalid confidence data type"))
                return False

            return True

        except Exception as e:
            self.logger.exception(error(f"Error validating strategy inputs: {e}"))
            return False

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error performing position sizing: {e}"))
            return {}

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error performing risk management: {e}"))
            return {}

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(
                error(f"Error performing portfolio optimization: {e}")
            )
            return {}

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error performing dynamic rebalancing: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Kelly Criterion: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Fixed Fraction: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Volatility Targeting: {e}"))
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

        except Exception as e:
            self.logger.exception(error(f"Error calculating Risk Parity: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Stop Loss: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Take Profit: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Trailing Stop: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Position Limits: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Mean Variance: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Black Litterman: {e}"))
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
        except Exception as e:
            self.logger.exception(
                error(f"Error calculating Portfolio Risk Parity: {e}")
            )
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Maximum Sharpe: {e}"))
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
        except Exception as e:
            self.logger.exception(
                error(f"Error calculating Threshold Rebalancing: {e}")
            )
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
            last_rebalance = datetime.now() - timedelta(days = 30)
            rebalance_interval = timedelta(days = 7)  # Weekly rebalancing

            return (current_time - last_rebalance) > rebalance_interval
        except Exception as e:
            self.logger.exception(error(f"Error calculating Calendar Rebalancing: {e}"))
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
        except Exception as e:
            self.logger.exception(error(f"Error calculating Drift Rebalancing: {e}"))
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
        except Exception as e:
            self.logger.exception(
                error(f"Error calculating Volatility Rebalancing: {e}")
            )
            return False

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error storing strategy results: {e}"))

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error getting strategy results: {e}"))
            return {}

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error getting strategy history: {e}"))
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

    @handles_errors(fallback = None)
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

        except Exception as e:
            self.logger.exception(error(f"Error stopping modular strategist: {e}"))

    # IStrategist interface implementation

    async def start(self) -> None:
        """Start the strategist (IStrategist interface)."""
        await self.initialize()

    async def formulate_strategy(self, analysis_result: AnalysisResult) -> StrategyResult:
        """Formulate trading strategy based on analysis (IStrategist interface)."""
        try:
            # Convert AnalysisResult to dict format for existing method
            market_data_dict = {
                "symbol": analysis_result.symbol,
                "price": 100.0,  # Default price for strategy formulation
                "volume": 1000.0,  # Default volume
                "timestamp": analysis_result.timestamp.isoformat(),
            }
            
            analysis_data_dict = {
                "signal": analysis_result.signal,
                "confidence": analysis_result.confidence,
                "technical": analysis_result.technical_indicators,
                "fundamental": analysis_result.features,
                "risk": analysis_result.risk_metrics
            }
            
            # Execute strategy using existing method
            success = await self.execute_strategy(market_data_dict, analysis_data_dict)
            
            if not success:
                # Return default strategy if execution failed
                return StrategyResult(
                    timestamp=analysis_result.timestamp,
                    symbol=analysis_result.symbol,
                    position_bias="NEUTRAL",
                    leverage_cap=1.0,
                    max_notional_size=1000.0,
                    risk_parameters={},
                    market_conditions={}
                )
            
            # Extract strategy results
            position_sizing = self.strategy_results.get("position_sizing", {})
            risk_management = self.strategy_results.get("risk_management", {})
            
            # Determine position bias based on analysis signal
            position_bias = self._determine_position_bias(analysis_result.signal, analysis_result.confidence)
            
            # Calculate leverage cap based on risk parameters
            leverage_cap = self._calculate_leverage_cap(risk_management, analysis_result.risk_metrics)
            
            # Calculate max notional size based on position sizing
            max_notional_size = self._calculate_max_notional_size(position_sizing, analysis_result.confidence)
            
            return StrategyResult(
                timestamp=analysis_result.timestamp,
                symbol=analysis_result.symbol,
                position_bias=position_bias,
                leverage_cap=leverage_cap,
                max_notional_size=max_notional_size,
                risk_parameters=risk_management,
                market_conditions={
                    "market_regime": analysis_result.market_regime,
                    "volatility": analysis_result.risk_metrics.get("volatility", 0.0),
                    "confidence": analysis_result.confidence
                }
            )
            
        except Exception as e:
            self.logger.exception(error(f"Error in formulate_strategy interface method: {e}"))
            return StrategyResult(
                timestamp=analysis_result.timestamp,
                symbol=analysis_result.symbol,
                position_bias="NEUTRAL",
                leverage_cap=1.0,
                max_notional_size=1000.0,
                risk_parameters={},
                market_conditions={}
            )

    async def update_strategy_parameters(self, parameters: dict[str, Any]) -> None:
        """Update strategy parameters (IStrategist interface)."""
        try:
            self.logger.info("Updating strategy parameters...")
            
            # Update configuration with new parameters
            if "strategy_interval" in parameters:
                self.strategy_interval = parameters["strategy_interval"]
                self.strategist_config["strategy_interval"] = parameters["strategy_interval"]
            
            if "enable_position_sizing" in parameters:
                self.enable_position_sizing = parameters["enable_position_sizing"]
                self.strategist_config["enable_position_sizing"] = parameters["enable_position_sizing"]
            
            if "enable_risk_management" in parameters:
                self.enable_risk_management = parameters["enable_risk_management"]
                self.strategist_config["enable_risk_management"] = parameters["enable_risk_management"]
            
            # Update other strategy-specific parameters
            for key, value in parameters.items():
                if key not in ["strategy_interval", "enable_position_sizing", "enable_risk_management"]:
                    self.strategist_config[key] = value
            
            self.logger.info("✅ Strategy parameters updated successfully")
            
        except Exception as e:
            self.logger.exception(error(f"Error updating strategy parameters: {e}"))

    async def get_strategy_performance(self) -> dict[str, Any]:
        """Get strategy performance metrics (IStrategist interface)."""
        try:
            # Calculate performance metrics from strategy history
            if not self.strategy_history:
                return {
                    "total_strategies": 0,
                    "success_rate": 0.0,
                    "avg_confidence": 0.0,
                    "risk_metrics": {},
                    "performance_score": 0.0
                }
            
            # Calculate basic metrics
            total_strategies = len(self.strategy_history)
            successful_strategies = sum(1 for strategy in self.strategy_history 
                                      if strategy.get("success", False))
            success_rate = successful_strategies / total_strategies if total_strategies > 0 else 0.0
            
            # Calculate average confidence
            confidences = [strategy.get("confidence", 0.0) for strategy in self.strategy_history]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate risk metrics
            risk_metrics = {}
            for strategy in self.strategy_history:
                risk_data = strategy.get("risk_management", {})
                for key, value in risk_data.items():
                    if key not in risk_metrics:
                        risk_metrics[key] = []
                    if isinstance(value, (int, float)):
                        risk_metrics[key].append(value)
            
            # Calculate averages for risk metrics
            avg_risk_metrics = {}
            for key, values in risk_metrics.items():
                if values:
                    avg_risk_metrics[key] = sum(values) / len(values)
            
            # Calculate performance score
            performance_score = (success_rate * 0.4 + avg_confidence * 0.6)
            
            return {
                "total_strategies": total_strategies,
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "risk_metrics": avg_risk_metrics,
                "performance_score": performance_score,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(error(f"Error getting strategy performance: {e}"))
            return {
                "total_strategies": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "risk_metrics": {},
                "performance_score": 0.0
            }

    # Helper methods for interface implementation

    def _determine_position_bias(self, signal: str, confidence: float) -> str:
        """Determine position bias based on signal and confidence."""
        try:
            if signal == "BUY" and confidence > 0.7:
                return "BULLISH"
            elif signal == "SELL" and confidence > 0.7:
                return "BEARISH"
            elif signal == "BUY" and confidence > 0.5:
                return "SLIGHTLY_BULLISH"
            elif signal == "SELL" and confidence > 0.5:
                return "SLIGHTLY_BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.exception(error(f"Error determining position bias: {e}"))
            return "NEUTRAL"

    def _calculate_leverage_cap(self, risk_management: dict, risk_metrics: dict) -> float:
        """Calculate leverage cap based on risk parameters."""
        try:
            base_leverage = 1.0
            
            # Adjust based on risk management parameters
            if "position_limits" in risk_management:
                position_limits = risk_management["position_limits"]
                if isinstance(position_limits, dict):
                    max_leverage = position_limits.get("max_leverage", 1.0)
                    base_leverage = min(base_leverage, max_leverage)
            
            # Adjust based on risk metrics
            volatility = risk_metrics.get("volatility", 0.0)
            if volatility > 0.3:  # High volatility
                base_leverage *= 0.5
            elif volatility > 0.2:  # Medium volatility
                base_leverage *= 0.75
            
            return max(min(base_leverage, 3.0), 0.1)  # Cap between 0.1 and 3.0
            
        except Exception as e:
            self.logger.exception(error(f"Error calculating leverage cap: {e}"))
            return 1.0

    def _calculate_max_notional_size(self, position_sizing: dict, confidence: float) -> float:
        """Calculate max notional size based on position sizing and confidence."""
        try:
            base_size = 1000.0
            
            # Adjust based on position sizing strategies
            if "kelly_criterion" in position_sizing:
                kelly_fraction = position_sizing["kelly_criterion"]
                if isinstance(kelly_fraction, (int, float)):
                    base_size *= kelly_fraction
            
            if "fixed_fraction" in position_sizing:
                fixed_fraction = position_sizing["fixed_fraction"]
                if isinstance(fixed_fraction, (int, float)):
                    base_size *= fixed_fraction
            
            # Adjust based on confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
            base_size *= confidence_multiplier
            
            return max(min(base_size, 10000.0), 100.0)  # Cap between 100 and 10000
            
        except Exception as e:
            self.logger.exception(error(f"Error calculating max notional size: {e}"))
            return 1000.0

# Global modular strategist instance
modular_strategist: ModularStrategist | None = None

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
        tprint(f"Error setting up modular strategist: {e}")
        return None
