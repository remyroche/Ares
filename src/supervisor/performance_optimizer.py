# src/supervisor/performance_optimizer.py

"""
Performance Optimizer for continuous financial performance improvement.
Implements adaptive parameter tuning, regime detection, and dynamic strategy adjustment.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import optuna

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class PerformanceOptimizer:
    """
    Advanced performance optimizer that continuously monitors and optimizes
    trading performance through adaptive parameter tuning and regime detection.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("PerformanceOptimizer")

        # Configuration
        self.optimizer_config = config.get("performance_optimizer", {})
        self.optimization_interval = self.optimizer_config.get(
            "optimization_interval",
            3600,
        )  # 1 hour
        self.performance_window = self.optimizer_config.get(
            "performance_window",
            24,
        )  # 24 hours
        self.min_trades_for_optimization = self.optimizer_config.get(
            "min_trades_for_optimization",
            10,
        )

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.parameter_history: list[dict[str, Any]] = []
        self.optimization_results: list[dict[str, Any]] = []

        # Optimization components
        self.regime_detector = None
        self.parameter_optimizer = None
        self.strategy_adapter = None

        # Performance metrics
        self.current_metrics = {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "calmar_ratio": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
        }

        self.is_initialized = False
        self.is_running = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance optimizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize performance optimizer components."""
        try:
            self.logger.info("🚀 Initializing performance optimizer...")

            # Initialize regime detector
            self.regime_detector = MarketRegimeDetector(self.config)
            await self.regime_detector.initialize()

            # Initialize parameter optimizer
            self.parameter_optimizer = ParameterOptimizer(self.config)
            await self.parameter_optimizer.initialize()

            # Initialize strategy adapter
            self.strategy_adapter = StrategyAdapter(self.config)
            await self.strategy_adapter.initialize()

            self.is_initialized = True
            self.logger.info("✅ Performance optimizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing performance optimizer: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance optimization",
    )
    async def optimize_performance(self, trading_data: dict[str, Any]) -> bool:
        """
        Optimize trading performance based on recent data.

        Args:
            trading_data: Dictionary containing trading performance data

        Returns:
            bool: True if optimization was successful
        """
        try:
            if not self.is_initialized:
                self.logger.error("Performance optimizer not initialized")
                return False

            # Update performance metrics
            await self._update_performance_metrics(trading_data)

            # Detect market regime
            regime_info = await self.regime_detector.detect_regime(trading_data)

            # Check if optimization is needed
            if not await self._should_optimize(trading_data):
                self.logger.info("⏭️ Skipping optimization - conditions not met")
                return True

            # Perform parameter optimization
            optimization_result = await self.parameter_optimizer.optimize_parameters(
                trading_data,
                regime_info,
            )

            if optimization_result:
                # Adapt strategy based on optimization results
                adaptation_result = await self.strategy_adapter.adapt_strategy(
                    optimization_result,
                    regime_info,
                )

                # Store optimization results
                self.optimization_results.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "regime": regime_info.get("market_regime", "unknown"),
                        "optimization_result": optimization_result,
                        "adaptation_result": adaptation_result,
                        "performance_metrics": self.current_metrics.copy(),
                    },
                )

                self.logger.info("✅ Performance optimization completed successfully")
                return True
            self.logger.warning("⚠️ Parameter optimization failed")
            return False

        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")
            return False

    async def _update_performance_metrics(self, trading_data: dict[str, Any]) -> None:
        """Update current performance metrics."""
        try:
            # Extract performance data
            trades = trading_data.get("trades", [])
            equity_curve = trading_data.get("equity_curve", [])

            if not trades or len(trades) < 5:
                return

            # Calculate performance metrics
            returns = self._calculate_returns(equity_curve)

            if len(returns) > 0:
                self.current_metrics.update(
                    {
                        "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                        "sortino_ratio": self._calculate_sortino_ratio(returns),
                        "max_drawdown": self._calculate_max_drawdown(equity_curve),
                        "win_rate": self._calculate_win_rate(trades),
                        "profit_factor": self._calculate_profit_factor(trades),
                        "calmar_ratio": self._calculate_calmar_ratio(returns),
                        "total_return": self._calculate_total_return(equity_curve),
                        "volatility": np.std(returns) if len(returns) > 0 else 0.0,
                    },
                )

            # Store performance history
            self.performance_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self.current_metrics.copy(),
                    "trade_count": len(trades),
                },
            )

            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _should_optimize(self, trading_data: dict[str, Any]) -> bool:
        """Determine if optimization is needed."""
        try:
            trades = trading_data.get("trades", [])

            # Check minimum trade count
            if len(trades) < self.min_trades_for_optimization:
                return False

            # Check performance degradation
            if len(self.performance_history) >= 5:
                recent_sharpe = np.mean(
                    [
                        h["metrics"]["sharpe_ratio"]
                        for h in self.performance_history[-5:]
                    ],
                )
                current_sharpe = self.current_metrics["sharpe_ratio"]

                # Optimize if Sharpe ratio has degraded significantly
                if current_sharpe < recent_sharpe * 0.9:
                    self.logger.info(
                        f"📉 Performance degradation detected: {current_sharpe:.3f} vs {recent_sharpe:.3f}",
                    )
                    return True

            # Check for regime change
            if len(self.performance_history) >= 10:
                regime_changed = await self._detect_regime_change()
                if regime_changed:
                    self.logger.info("🔄 Market regime change detected")
                    return True

            # Periodic optimization (every 24 hours)
            if len(self.optimization_results) == 0:
                return True

            last_optimization = datetime.fromisoformat(
                self.optimization_results[-1]["timestamp"],
            )
            if datetime.now() - last_optimization > timedelta(hours=24):
                self.logger.info("⏰ Periodic optimization due")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking optimization conditions: {e}")
            return False

    async def _detect_regime_change(self) -> bool:
        """Detect if market regime has changed."""
        try:
            if len(self.performance_history) < 10:
                return False

            # Calculate regime stability metrics
            recent_volatility = np.mean(
                [h["metrics"]["volatility"] for h in self.performance_history[-5:]],
            )
            historical_volatility = np.mean(
                [h["metrics"]["volatility"] for h in self.performance_history[-10:-5]],
            )

            # Check for significant volatility change
            volatility_change = (
                abs(recent_volatility - historical_volatility) / historical_volatility
            )
            if volatility_change > 0.3:  # 30% change
                return True

            # Check for trend change
            recent_returns = np.mean(
                [h["metrics"]["total_return"] for h in self.performance_history[-5:]],
            )
            historical_returns = np.mean(
                [
                    h["metrics"]["total_return"]
                    for h in self.performance_history[-10:-5]
                ],
            )

            if (recent_returns > 0 and historical_returns < 0) or (
                recent_returns < 0 and historical_returns > 0
            ):
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error detecting regime change: {e}")
            return False

    def _calculate_returns(self, equity_curve: list[float]) -> list[float]:
        """Calculate returns from equity curve."""
        try:
            if len(equity_curve) < 2:
                return []

            returns = []
            for i in range(1, len(equity_curve)):
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)

            return returns
        except Exception:
            return []

    def _calculate_sharpe_ratio(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not returns:
                return 0.0

            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                return 0.0

            return mean_return / std_return * np.sqrt(252)  # Annualized
        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, returns: list[float]) -> float:
        """Calculate Sortino ratio."""
        try:
            if not returns:
                return 0.0

            mean_return = np.mean(returns)
            negative_returns = [r for r in returns if r < 0]

            if not negative_returns:
                return np.inf

            downside_std = np.std(negative_returns)

            if downside_std == 0:
                return 0.0

            return mean_return / downside_std * np.sqrt(252)  # Annualized
        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, equity_curve: list[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not equity_curve:
                return 0.0

            peak = equity_curve[0]
            max_dd = 0.0

            for value in equity_curve:
                peak = max(value, peak)
                dd = (peak - value) / peak
                max_dd = max(dd, max_dd)

            return max_dd
        except Exception:
            return 0.0

    def _calculate_win_rate(self, trades: list[dict[str, Any]]) -> float:
        """Calculate win rate."""
        try:
            if not trades:
                return 0.0

            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            return len(winning_trades) / len(trades)
        except Exception:
            return 0.0

    def _calculate_profit_factor(self, trades: list[dict[str, Any]]) -> float:
        """Calculate profit factor."""
        try:
            if not trades:
                return 0.0

            gross_profit = sum([t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0])
            gross_loss = abs(
                sum([t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]),
            )

            if gross_loss == 0:
                return np.inf if gross_profit > 0 else 0.0

            return gross_profit / gross_loss
        except Exception:
            return 0.0

    def _calculate_calmar_ratio(self, returns: list[float]) -> float:
        """Calculate Calmar ratio."""
        try:
            if not returns:
                return 0.0

            total_return = np.prod([1 + r for r in returns]) - 1
            max_dd = self._calculate_max_drawdown(
                [1 + sum(returns[: i + 1]) for i in range(len(returns))],
            )

            if max_dd == 0:
                return 0.0

            return total_return / max_dd
        except Exception:
            return 0.0

    def _calculate_total_return(self, equity_curve: list[float]) -> float:
        """Calculate total return."""
        try:
            if len(equity_curve) < 2:
                return 0.0

            return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        except Exception:
            return 0.0

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance optimization summary."""
        return {
            "current_metrics": self.current_metrics.copy(),
            "optimization_count": len(self.optimization_results),
            "last_optimization": self.optimization_results[-1]["timestamp"]
            if self.optimization_results
            else None,
            "performance_trend": self._calculate_performance_trend(),
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend."""
        try:
            if len(self.performance_history) < 10:
                return "insufficient_data"

            recent_sharpe = np.mean(
                [h["metrics"]["sharpe_ratio"] for h in self.performance_history[-5:]],
            )
            historical_sharpe = np.mean(
                [
                    h["metrics"]["sharpe_ratio"]
                    for h in self.performance_history[-10:-5]
                ],
            )

            if recent_sharpe > historical_sharpe * 1.1:
                return "improving"
            if recent_sharpe < historical_sharpe * 0.9:
                return "declining"
            return "stable"
        except Exception:
            return "unknown"


class MarketRegimeDetector:
    """Detect market regimes for optimization."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("MarketRegimeDetector")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize regime detector."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing regime detector: {e}")
            return False

    async def detect_regime(self, trading_data: dict[str, Any]) -> dict[str, Any]:
        """Detect market regime from trading data."""
        try:
            trades = trading_data.get("trades", [])

            if not trades:
                return {"market_regime": "unknown", "confidence": 0.0}

            # Calculate regime indicators
            volatility = self._calculate_volatility(trades)
            trend = self._calculate_trend(trades)
            volume_profile = self._calculate_volume_profile(trades)

            # Classify regime
            regime = self._classify_regime(volatility, trend, volume_profile)

            return {
                "market_regime": regime,
                "confidence": 0.8,
                "volatility": volatility,
                "trend": trend,
                "volume_profile": volume_profile,
            }

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return {"market_regime": "unknown", "confidence": 0.0}

    def _calculate_volatility(self, trades: list[dict[str, Any]]) -> float:
        """Calculate volatility from trades."""
        try:
            if len(trades) < 5:
                return 0.0

            pnls = [t.get("pnl", 0) for t in trades[-20:]]
            return np.std(pnls)
        except Exception:
            return 0.0

    def _calculate_trend(self, trades: list[dict[str, Any]]) -> str:
        """Calculate trend from trades."""
        try:
            if len(trades) < 10:
                return "neutral"

            recent_pnls = [t.get("pnl", 0) for t in trades[-5:]]
            historical_pnls = [t.get("pnl", 0) for t in trades[-10:-5]]

            recent_avg = np.mean(recent_pnls)
            historical_avg = np.mean(historical_pnls)

            if recent_avg > historical_avg * 1.2:
                return "bullish"
            if recent_avg < historical_avg * 0.8:
                return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    def _calculate_volume_profile(self, trades: list[dict[str, Any]]) -> str:
        """Calculate volume profile from trades."""
        try:
            if len(trades) < 5:
                return "normal"

            volumes = [t.get("volume", 0) for t in trades[-10:]]
            avg_volume = np.mean(volumes)
            current_volume = volumes[-1] if volumes else 0

            if current_volume > avg_volume * 1.5:
                return "high"
            if current_volume < avg_volume * 0.5:
                return "low"
            return "normal"
        except Exception:
            return "normal"

    def _classify_regime(
        self,
        volatility: float,
        trend: str,
        volume_profile: str,
    ) -> str:
        """Classify market regime."""
        if trend == "bullish" and volume_profile == "high":
            return "bull_high_vol" if volatility > 0.02 else "bull_low_vol"
        if trend == "bearish" and volume_profile == "high":
            return "bear_high_vol" if volatility > 0.02 else "bear_low_vol"
        if trend == "neutral":
            return "sideways"
        return "mixed"


class ParameterOptimizer:
    """Optimize trading parameters using machine learning."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("ParameterOptimizer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize parameter optimizer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing parameter optimizer: {e}")
            return False

    async def optimize_parameters(
        self,
        trading_data: dict[str, Any],
        regime_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Optimize trading parameters."""
        try:
            # Define optimization parameters
            param_ranges = {
                "entry_confidence_threshold": (0.5, 0.9),
                "stop_loss_threshold": (0.01, 0.05),
                "take_profit_threshold": (0.02, 0.08),
                "position_size_multiplier": (0.5, 2.0),
                "max_positions": (1, 5),
            }

            # Create optimization objective
            def objective(trial):
                params = {
                    "entry_confidence_threshold": trial.suggest_float(
                        "entry_confidence_threshold",
                        0.5,
                        0.9,
                    ),
                    "stop_loss_threshold": trial.suggest_float(
                        "stop_loss_threshold",
                        0.01,
                        0.05,
                    ),
                    "take_profit_threshold": trial.suggest_float(
                        "take_profit_threshold",
                        0.02,
                        0.08,
                    ),
                    "position_size_multiplier": trial.suggest_float(
                        "position_size_multiplier",
                        0.5,
                        2.0,
                    ),
                    "max_positions": trial.suggest_int("max_positions", 1, 5),
                }

                # Simulate performance with these parameters
                performance = self._simulate_performance(
                    trading_data,
                    params,
                    regime_info,
                )
                return performance

            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            best_params = study.best_params
            best_value = study.best_value

            return {
                "optimized_parameters": best_params,
                "expected_performance": best_value,
                "optimization_confidence": 0.8,
            }

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return {}

    def _simulate_performance(
        self,
        trading_data: dict[str, Any],
        params: dict[str, Any],
        regime_info: dict[str, Any],
    ) -> float:
        """Simulate performance with given parameters."""
        try:
            # Simple performance simulation based on parameter adjustments
            base_performance = 1.0

            # Adjust based on entry threshold
            entry_threshold = params["entry_confidence_threshold"]
            if entry_threshold > 0.7:
                base_performance *= 1.1  # Higher threshold = better quality trades
            elif entry_threshold < 0.6:
                base_performance *= 0.9  # Lower threshold = more noise

            # Adjust based on stop loss
            stop_loss = params["stop_loss_threshold"]
            if stop_loss < 0.02:
                base_performance *= 0.8  # Too tight stops
            elif stop_loss > 0.04:
                base_performance *= 0.9  # Too loose stops
            else:
                base_performance *= 1.0  # Optimal range

            # Adjust based on regime
            regime = regime_info.get("market_regime", "unknown")
            if regime == "bull_low_vol":
                base_performance *= 1.2
            elif regime == "bear_high_vol":
                base_performance *= 0.7
            elif regime == "sideways":
                base_performance *= 0.9

            return base_performance

        except Exception:
            return 0.0


class StrategyAdapter:
    """Adapt trading strategy based on optimization results."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("StrategyAdapter")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize strategy adapter."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing strategy adapter: {e}")
            return False

    async def adapt_strategy(
        self,
        optimization_result: dict[str, Any],
        regime_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Adapt trading strategy based on optimization results."""
        try:
            optimized_params = optimization_result.get("optimized_parameters", {})
            regime = regime_info.get("market_regime", "unknown")

            # Generate strategy adaptations
            adaptations = {
                "parameter_updates": optimized_params,
                "regime_specific_adjustments": self._get_regime_adjustments(regime),
                "risk_management_updates": self._get_risk_management_updates(
                    optimization_result,
                ),
                "position_sizing_updates": self._get_position_sizing_updates(
                    optimization_result,
                ),
            }

            self.logger.info(f"🔄 Strategy adapted for regime: {regime}")
            return adaptations

        except Exception as e:
            self.logger.error(f"Error adapting strategy: {e}")
            return {}

    def _get_regime_adjustments(self, regime: str) -> dict[str, Any]:
        """Get regime-specific strategy adjustments."""
        adjustments = {
            "bull_low_vol": {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 1.3,
            },
            "bull_high_vol": {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 1.1,
            },
            "bear_low_vol": {
                "position_size_multiplier": 0.6,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 0.9,
            },
            "bear_high_vol": {
                "position_size_multiplier": 0.4,
                "stop_loss_multiplier": 0.6,
                "take_profit_multiplier": 0.7,
            },
            "sideways": {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
            },
        }

        return adjustments.get(regime, adjustments["sideways"])

    def _get_risk_management_updates(
        self,
        optimization_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Get risk management updates."""
        return {
            "max_portfolio_risk": 0.02,
            "max_position_risk": 0.01,
            "correlation_limit": 0.7,
            "sector_allocation_limit": 0.3,
        }

    def _get_position_sizing_updates(
        self,
        optimization_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Get position sizing updates."""
        return {
            "kelly_criterion_enabled": True,
            "volatility_targeting": True,
            "dynamic_sizing": True,
            "confidence_based_sizing": True,
        }
