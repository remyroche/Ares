#!/usr/bin/env python3
"""
Enhanced Backtester with Comprehensive Reporting

This module provides enhanced backtesting capabilities with detailed reporting
that matches the paper trading metrics for consistency across all trading modes.
"""


from datetime import datetime
from typing import Any, TYPE_CHECKING
import json
import os

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
import error,
	error,
	failed,
	initialization_error,
	invalid,
	warning,
)
from src.utils.trading_decorators import ExecutionMode, get_trade_tracker

import if TYPE_CHECKING:  # Avoid importing potentially missing modules at runtime
if TYPE_CHECKING:  # Avoid importing potentially missing modules at runtime
	# Only for type hints
	from src.reports.paper_trading_reporter import PaperTradingReporter  # pragma: no cover


import class EnhancedBacktester:
class EnhancedBacktester:
	"""
	Enhanced backtester with comprehensive reporting capabilities.
	"""

	def __init__(self, config: dict[str, Any]) -> None:
    pass
    pass
		"""
		Initialize enhanced backtester.

		Args:
			config: Configuration dictionary
		"""
		self.config = config
		self.logger = system_logger.getChild("EnhancedBacktester")

		# Backtesting state
		self.is_running: bool = False
		self.current_position: dict[str, Any] = {}
		self.trade_history: list[dict[str, Any]] = []
		self.portfolio_value: float = 10000.0
		self.initial_balance: float = 10000.0

		# Configuration
		self.backtest_config = config.get("enhanced_backtester", {})
		self.initial_balance = float(self.backtest_config.get("initial_balance", 10000.0))
		self.commission_rate = float(self.backtest_config.get("commission_rate", 0.001))
		self.slippage_rate = float(self.backtest_config.get("slippage_rate", 0.0005))
		self.max_position_size = float(self.backtest_config.get("max_position_size", 0.1))

		# Enhanced reporting
		self.reporter: PaperTradingReporter | None = None
		self.enable_detailed_reporting = bool(
			self.backtest_config.get("enable_detailed_reporting", True),
		)

		# Performance tracking
		self.performance_metrics: dict[str, Any] = {}
		self.equity_curve: list[float] = []
		self.drawdown_curve: list[float] = []

		# Trade tracking
		self.trade_tracker = get_trade_tracker()

	@handle_specific_errors(
		error_handlers={
			ValueError: (False, "Invalid backtester configuration"),
			AttributeError: (False, "Missing required backtester parameters"),
		},
		default_return=False,
		context="backtester initialization",
	)
	async def initialize(self) -> bool:
		"""
		Initialize enhanced backtester with reporting capabilities.

		Returns:
			bool: True if initialization successful = False otherwise
		"""
		try:
			self.logger.info("Initializing Enhanced Backtester...")

    except Exception as e:
        pass
    except Exception as e:
        pass
			# Load backtester configuration
			await self._load_backtester_configuration()

			# Validate configuration
			if not self._validate_configuration():
    pass
    pass
				self.logger.error(invalid("Invalid configuration for enhanced backtester"))
				return False

			# Initialize backtesting state
			await self._initialize_backtesting_state()

			# Initialize detailed reporting
			if self.enable_detailed_reporting:
    pass
    pass
				await self._initialize_detailed_reporting()

			self.logger.info(
				"✅ Enhanced Backtester initialization completed successfully",
			)
			return True

		except Exception as e:  # pragma: no cover - safety
			self.logger.error(failed(f"❌ Enhanced Backtester initialization failed: {e}"))
			return False

	@handle_errors(
		exceptions=(ValueError, AttributeError),
		default_return=None,
		context="backtester configuration loading",
	)
	async def _load_backtester_configuration(self) -> None:
		"""Load backtester configuration."""
		try:
			# Set default parameters
    except Exception as e:
        pass
    except Exception as e:
        pass
			self.backtest_config.setdefault("initial_balance", 10000.0)
			self.backtest_config.setdefault("commission_rate", 0.001)
			self.backtest_config.setdefault("slippage_rate", 0.0005)
			self.backtest_config.setdefault("max_position_size", 0.1)
			self.backtest_config.setdefault("enable_detailed_reporting", True)

			# Update configuration
			self.initial_balance = float(self.backtest_config["initial_balance"])
			self.commission_rate = float(self.backtest_config["commission_rate"])
			self.slippage_rate = float(self.backtest_config["slippage_rate"])
			self.max_position_size = float(self.backtest_config["max_position_size"])
			self.enable_detailed_reporting = bool(
				self.backtest_config["enable_detailed_reporting"],
			)

		except Exception as e:  # pragma: no cover - safety
			self.logger.error(error(f"Error loading backtester configuration: {e}"))

	@handle_errors(
		exceptions=(ValueError, AttributeError),
		default_return=False,
		context="configuration validation",
	)
	def _validate_configuration(self) -> bool:
    pass
    pass
		"""Validate backtester configuration."""
		try:
			if self.initial_balance <= 0:
    pass
    pass
				self.logger.error(initialization_error("Initial balance must be positive"))
				return False

    except Exception as e:
        pass
    except Exception as e:
        pass
			if self.commission_rate < 0 or self.commission_rate > 0.1:
    pass
    pass
				self.logger.error(error("Commission rate must be between 0 and 0.1"))
				return False

			if self.slippage_rate < 0 or self.slippage_rate > 0.1:
    pass
    pass
				self.logger.error(error("Slippage rate must be between 0 and 0.1"))
				return False

			if self.max_position_size <= 0 or self.max_position_size > 1.0:
    pass
    pass
				self.logger.error(error("Max position size must be between 0 and 1"))
				return False

			return True

		except Exception as e:  # pragma: no cover - safety
			self.logger.error(error(f"Error validating configuration: {e}"))
			return False

	@handle_errors(
		exceptions=(ValueError, AttributeError),
		default_return=None,
		context="backtesting state initialization",
	)
	async def _initialize_backtesting_state(self) -> None:
		"""Initialize backtesting state."""
		try:
			self.portfolio_value = float(self.initial_balance)
    except Exception as e:
        pass
    except Exception as e:
        pass
			self.current_position = {}
			self.trade_history = []
			self.equity_curve = [self.initial_balance]
			self.drawdown_curve = [0.0]

			self.logger.info(
				f"✅ Backtesting state initialized with balance: ${self.portfolio_value:.2f}",
			)

		except Exception as e:  # pragma: no cover - safety
			self.logger.error(
				initialization_error(f"Error initializing backtesting state: {e}"),
			)

	@handle_errors(
		exceptions=(ValueError, AttributeError),
		default_return=None,
		context="detailed reporting initialization",
	)
	async def _initialize_detailed_reporting(self) -> None:
		"""Initialize detailed reporting system."""
		try:
			# Import lazily to avoid hard dependency when reporter is not available
    except Exception as e:
        pass
    except Exception as e:
        pass
			try:
				from src.reports.paper_trading_reporter import (
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import setup_paper_trading_reporter as _setup_reporter,
					setup_paper_trading_reporter as _setup_reporter,
				)
			except Exception as e:  # pragma: no cover - safety
				self.logger.warning(
					warning(
						f"Detailed reporter unavailable, continuing without it: {e}",
					),
				)
				self.enable_detailed_reporting = False
				return

			self.reporter = await _setup_reporter(self.config)
			if self.reporter:
    pass
    pass
				self.logger.info("✅ Detailed reporting initialized successfully")
			else:
				self.logger.warning(
					warning("Failed to initialize detailed reporting; continuing"),
				)
				self.enable_detailed_reporting = False

		except Exception as e:  # pragma: no cover - safety
			self.logger.error(
				initialization_error(f"Error initializing detailed reporting: {e}"),
			)

	@handle_errors(
		exceptions=(Exception,),
		default_return={},
		context="backtest run",
	)
	async def run_backtest(
		self,
		historical_data: pd.DataFrame,
		strategy_signals: pd.DataFrame,
		trade_metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Run enhanced backtest with comprehensive reporting.

		Args:
			historical_data: Historical market data
			strategy_signals: Strategy signals DataFrame
			trade_metadata: Additional trade metadata

		Returns:
			Dict[str, Any]: Backtest results with detailed metrics
		"""
		self.logger.info("Starting enhanced backtest...")
		self.is_running = True

		# Initialize results
		results: dict[str, Any] = {
			"trades": [],
			"performance_metrics": {},
			"equity_curve": [],
			"drawdown_curve": [],
			"detailed_analysis": {},
		}

		# Process each signal
		if trade_metadata is None:
    pass
    pass
			trade_metadata = {}

		for index, row in strategy_signals.iterrows():
    pass
    pass
			if not self.is_running:
    pass
    pass
				break

			timestamp = row.name if hasattr(row.name, "isoformat") else pd.Timestamp(index)
			signal = int(row.get("signal", 0))  # 1 for buy, -1 for sell, 0 for hold
			price = float(row.get("close", 0))
			symbol = str(row.get("symbol", "UNKNOWN"))

			if signal != 0 and price > 0:
    pass
    pass
				trade_result = await self._execute_backtest_trade(
					symbol=symbol,
					signal=signal,
					price=price,
					timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
					trade_metadata=trade_metadata,
				)

				if trade_result:
    pass
    pass
					results["trades"].append(trade_result)
					# Optional: also log to dedicated backtest log if available
					try:  # pragma: no cover - best-effort logging
						from src.utils.comprehensive_logger import get_comprehensive_logger

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import cl = get_comprehensive_logger
						cl = get_comprehensive_logger()
						if cl:
    pass
    pass
							cl.log_backtest(
								f"TRADE {trade_result.get('side')} {trade_result.get('quantity', 0):.6f} {symbol} @ ${price:.4f} ts={timestamp.isoformat()}",
							)
					except Exception:
						pass

			# Update equity curve per iteration
			self._update_equity_curve()

		# Calculate final performance metrics
		results["performance_metrics"] = self._calculate_performance_metrics()
		results["equity_curve"] = self.equity_curve.copy()
		results["drawdown_curve"] = self.drawdown_curve.copy()

		# Generate detailed analysis if available
		if self.reporter:
    pass
    pass
			try:
				results["detailed_analysis"] = await self._generate_detailed_analysis()
    except Exception as e:
        pass
    except Exception as e:
        pass
			except Exception:
				results["detailed_analysis"] = {}

		self.logger.info("✅ Enhanced backtest completed successfully")
		try:  # pragma: no cover - best-effort logging
			from src.utils.comprehensive_logger import get_comprehensive_logger

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import cl = get_comprehensive_logger
			cl = get_comprehensive_logger()
			if cl:
    pass
    pass
				cl.log_backtest("Backtest completed successfully")
		except Exception:
			pass

		return results

	async def _execute_backtest_trade(
		self,
		symbol: str,
		signal: int,
		price: float,
		timestamp: datetime,
		trade_metadata: dict[str, Any] | None = None,
	) -> dict[str, Any] | None:
		"""Execute a trade during backtesting."""
		if trade_metadata is None:
    pass
    pass
			trade_metadata = {}

		# Extract trade metadata for tracking (optional fields)
		model_weights = trade_metadata.get("model_weights", {})
		model_confidences = trade_metadata.get("model_confidences", {})
		regime_analysis = trade_metadata.get("regime_analysis", {})
		hmm_regime = trade_metadata.get("hmm_regime", "")
		support_resistance = trade_metadata.get("support_resistance_levels", {})
		market_conditions = trade_metadata.get("market_conditions", {})
		risk_metrics = trade_metadata.get("risk_metrics", {})

		# Position sizing
		position_size = float(self.portfolio_value * self.max_position_size)
		if price <= 0:
    pass
    pass
			return None
		quantity = position_size / price

		if signal == 1:  # Buy
			return await self._execute_buy_trade(
				symbol=symbol,
				quantity=quantity,
				price=price,
				timestamp=timestamp,
				model_weights=model_weights,
				model_confidences=model_confidences,
				regime_analysis=regime_analysis,
				hmm_regime=hmm_regime,
				support_resistance_levels=support_resistance,
				market_conditions=market_conditions,
				risk_metrics=risk_metrics,
			)
		if signal == -1:  # Sell
			return await self._execute_sell_trade(
				symbol=symbol,
				quantity=quantity,
				price=price,
				timestamp=timestamp,
				model_weights=model_weights,
				model_confidences=model_confidences,
				regime_analysis=regime_analysis,
				hmm_regime=hmm_regime,
				support_resistance_levels=support_resistance,
				market_conditions=market_conditions,
				risk_metrics=risk_metrics,
			)
		return None

	async def _execute_buy_trade(
		self,
		symbol: str,
		quantity: float,
		price: float,
		timestamp: datetime,
		model_weights: dict[str, float],
		model_confidences: dict[str, float],
		regime_analysis: dict[str, Any],
		hmm_regime: str,
		support_resistance_levels: dict[str, float],
		market_conditions: dict[str, Any],
		risk_metrics: dict[str, float],
	) -> dict[str, Any] | None:
		"""Execute a buy trade during backtesting."""
		# Calculate costs
		total_cost = float(quantity * price)
		commission = float(total_cost * self.commission_rate)
		slippage = float(total_cost * self.slippage_rate)
		total_with_fees = total_cost + commission + slippage

		# Check balance
		if total_with_fees > self.portfolio_value:
    pass
    pass
			self.logger.warning(
				f"Insufficient balance for buy trade: ${total_with_fees:.2f} > ${self.portfolio_value:.2f}",
			)
			return None

		# Execute the trade: deduct cash
		self.portfolio_value -= total_with_fees

		# Update position aggregate
		if symbol not in self.current_position:
    pass
    pass
			self.current_position[symbol] = {
				"quantity": 0.0,
				"avg_price": 0.0,
				"total_cost": 0.0,
			}

		position = self.current_position[symbol]
		old_quantity = float(position["quantity"])  # before trade
		old_total_cost = float(position["total_cost"])  # before trade

		new_quantity = old_quantity + quantity
		new_total_cost = old_total_cost + total_cost
		new_avg_price = new_total_cost / new_quantity if new_quantity > 0 else 0.0

		position["quantity"] = new_quantity
		position["avg_price"] = new_avg_price
		position["total_cost"] = new_total_cost

		# Create trade record
		trade_id = f"BUY_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
		trade_record: dict[str, Any] = {
			"trade_id": trade_id,
			"timestamp": timestamp,
			"symbol": symbol,
			"side": "BUY",
			"quantity": quantity,
			"price": price,
			"total_cost": total_cost,
			"commission": commission,
			"slippage": slippage,
			"portfolio_value_after": self.portfolio_value,
			"execution_mode": ExecutionMode.BACKTEST.value,
			"model_weights": model_weights,
			"model_confidences": model_confidences,
			"regime_analysis": regime_analysis,
			"hmm_regime": hmm_regime,
			"support_resistance_levels": support_resistance_levels,
			"market_conditions": market_conditions,
			"risk_metrics": risk_metrics,
		}
		self.trade_history.append(trade_record)

		# Record detailed trade if reporting is enabled
		if self.enable_detailed_reporting and self.reporter:
    pass
    pass
			await self._record_detailed_backtest_trade(
				symbol=symbol,
				side="long",
				quantity=quantity,
				price=price,
				timestamp=timestamp,
				trade_metadata={
					"model_weights": model_weights,
					"model_confidences": model_confidences,
					"regime_analysis": regime_analysis,
					"hmm_regime": hmm_regime,
					"support_resistance_levels": support_resistance_levels,
					"market_conditions": market_conditions,
					"risk_metrics": risk_metrics,
				},
				total_cost=total_cost,
				commission=commission,
				slippage=slippage,
			)

		return trade_record

	async def _execute_sell_trade(
		self,
		symbol: str,
		quantity: float,
		price: float,
		timestamp: datetime,
		model_weights: dict[str, float],
		model_confidences: dict[str, float],
		regime_analysis: dict[str, Any],
		hmm_regime: str,
		support_resistance_levels: dict[str, float],
		market_conditions: dict[str, Any],
		risk_metrics: dict[str, float],
	) -> dict[str, Any] | None:
		"""Execute a sell trade during backtesting."""
		# Check if we have enough position
		if (
			symbol not in self.current_position
			or self.current_position[symbol]["quantity"] < quantity
		):
			self.logger.warning(
				f"Insufficient position for sell trade: {quantity} > {self.current_position.get(symbol, {}).get('quantity', 0)}",
			)
			return None

		# Snapshot current position for PnL calc
		position = self.current_position[symbol]
		old_quantity = float(position["quantity"])  # before trade
		old_total_cost = float(position["total_cost"])  # before trade
		old_avg_price = float(position.get("avg_price", 0.0))

		# Calculate proceeds
		total_proceeds = float(quantity * price)
		commission = float(total_proceeds * self.commission_rate)
		slippage = float(total_proceeds * self.slippage_rate)
		net_proceeds = total_proceeds - commission - slippage

		# Execute the trade: add cash
		self.portfolio_value += net_proceeds

		# Update position after selling
		new_quantity = old_quantity - quantity
		if new_quantity > 0:
    pass
    pass
			remaining_ratio = new_quantity / old_quantity
			new_total_cost = old_total_cost * remaining_ratio
			new_avg_price = new_total_cost / new_quantity if new_quantity > 0 else 0.0
		else:
			new_total_cost = 0.0
			new_avg_price = 0.0

		position["quantity"] = new_quantity
		position["avg_price"] = new_avg_price
		position["total_cost"] = new_total_cost

		# Remove position if fully closed
		if new_quantity <= 0:
    pass
    pass
			del self.current_position[symbol]

		# Calculate PnL versus average entry cost of the portion sold
		pnl = float(net_proceeds - (quantity * old_avg_price))
		pnl_percentage = float((pnl / (quantity * old_avg_price)) * 100) if old_avg_price > 0 else 0.0

		# Create trade record
		trade_id = f"SELL_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
		trade_record: dict[str, Any] = {
			"trade_id": trade_id,
			"timestamp": timestamp,
			"symbol": symbol,
			"side": "SELL",
			"quantity": quantity,
			"price": price,
			"total_proceeds": total_proceeds,
			"commission": commission,
			"slippage": slippage,
			"net_proceeds": net_proceeds,
			"pnl": pnl,
			"pnl_percentage": pnl_percentage,
			"portfolio_value_after": self.portfolio_value,
			"execution_mode": ExecutionMode.BACKTEST.value,
			"model_weights": model_weights,
			"model_confidences": model_confidences,
			"regime_analysis": regime_analysis,
			"hmm_regime": hmm_regime,
			"support_resistance_levels": support_resistance_levels,
			"market_conditions": market_conditions,
			"risk_metrics": risk_metrics,
		}
		self.trade_history.append(trade_record)

		# Record detailed trade if reporting is enabled
		if self.enable_detailed_reporting and self.reporter:
    pass
    pass
			await self._record_detailed_backtest_trade(
				symbol=symbol,
				side="short",
				quantity=quantity,
				price=price,
				timestamp=timestamp,
				trade_metadata=trade_metadata if trade_metadata is not None else {},
				total_proceeds=total_proceeds,
				net_proceeds=net_proceeds,
				pnl=pnl,
				pnl_percentage=pnl_percentage,
				commission=commission,
				slippage=slippage,
			)

		return trade_record

	@handle_errors(
		exceptions=(Exception,),
		default_return=None,
		context="detailed backtest trade recording",
	)
	async def _record_detailed_backtest_trade(
		self,
		symbol: str,
		side: str,
		quantity: float,
		price: float,
		timestamp: datetime,
		trade_metadata: dict[str, Any],
		**kwargs: Any,
	) -> None:
		"""Record detailed backtest trade information."""
		if not self.reporter:
    pass
    pass
			return

		# Prepare trade data structure
		trade_data: dict[str, Any] = {
			"symbol": symbol,
			"side": side,
			"quantity": quantity,
			"price": price,
			"timestamp": timestamp.isoformat(),
			"exchange": "backtest",
			"leverage": trade_metadata.get("leverage", 1.0),
			"duration": trade_metadata.get("duration", "backtest"),
			"strategy": trade_metadata.get("strategy", "backtest_strategy"),
			"order_type": trade_metadata.get("order_type", "market"),
			"portfolio_percentage": trade_metadata.get("portfolio_percentage", 0.0),
			"risk_percentage": trade_metadata.get("risk_percentage", 0.0),
			"max_position_size": trade_metadata.get("max_position_size", 0.0),
			"position_ranking": trade_metadata.get("position_ranking", 0),
			"status": "closed" if side == "short" else "open",
			"execution_quality": trade_metadata.get("execution_quality", 0.0),
			"risk_metrics": trade_metadata.get("risk_metrics", {}),
			"notes": trade_metadata.get("notes"),
		}

		if side == "long":
    pass
    pass
			trade_data.update(
				{
					"total_cost": kwargs.get("total_cost", 0.0),
					"absolute_pnl": 0.0,
					"percentage_pnl": 0.0,
					"unrealized_pnl": 0.0,
					"realized_pnl": 0.0,
					"net_pnl": 0.0,
				},
			)
		else:
			trade_data.update(
				{
					"total_proceeds": kwargs.get("total_proceeds", 0.0),
					"net_proceeds": kwargs.get("net_proceeds", 0.0),
					"absolute_pnl": kwargs.get("pnl", 0.0),
					"percentage_pnl": kwargs.get("pnl_percentage", 0.0),
					"realized_pnl": kwargs.get("pnl", 0.0),
					"net_pnl": kwargs.get("pnl", 0.0),
				},
			)

		# Add commission and slippage
		trade_data.update(
			{"commission": kwargs.get("commission", 0.0), "slippage": kwargs.get("slippage", 0.0)},
		)

		# Best-effort reporter call; interface may vary
		try:  # pragma: no cover - integration surface may vary
			# Common interface: record_trade(trade_data, market_indicators=..., ml_confidence=...)
    except Exception as e:
        pass
    except Exception as e:
        pass
			market_indicators = trade_metadata.get("market_indicators", {})
			ml_confidence = trade_metadata.get("ml_confidence", {})
			await self.reporter.record_trade(trade_data, market_indicators=market_indicators, ml_confidence=ml_confidence)  # type: ignore[attr-defined]
		except Exception:
			# Swallow to avoid breaking backtest
			pass

	def _update_equity_curve(self) -> None:
    pass
    pass
		"""Update equity curve and drawdown."""
		# Calculate current portfolio value (cash + simplistic mark-to-market)
		current_value = float(self.portfolio_value)
		for position in self.current_position.values():
    pass
    pass
			if position["quantity"] > 0:
    pass
    pass
				# Simplified: assume current price equals avg_price for unrealized
				current_value += float(position["quantity"]) * float(position["avg_price"])

		self.equity_curve.append(current_value)

		# Calculate drawdown
		peak = max(self.equity_curve) if self.equity_curve else current_value
		current_drawdown = (peak - current_value) / peak if peak > 0 else 0.0
		self.drawdown_curve.append(float(current_drawdown))

	def _calculate_performance_metrics(self) -> dict[str, Any]:
    pass
    pass
		"""Calculate comprehensive performance metrics."""
		if not self.trade_history:
    pass
    pass
			return {
				"total_trades": 0,
				"win_rate": 0.0,
				"total_pnl": 0.0,
				"max_drawdown": 0.0,
				"sharpe_ratio": 0.0,
				"total_return": 0.0,
			}

		# Basic counts
		total_trades = len(self.trade_history)
		sell_trades = [t for t in self.trade_history if t.get("side") == "SELL"]

		# P&L aggregation
		total_pnl = float(sum(t.get("pnl", 0.0) for t in sell_trades))
		total_cost = float(sum(t.get("total_cost", 0.0) for t in self.trade_history if t.get("side") == "BUY"))
		total_proceeds = float(sum(t.get("net_proceeds", 0.0) for t in sell_trades))

		# Win rate
		profitable_trades = len([t for t in sell_trades if t.get("pnl", 0.0) > 0])
		win_rate = float(profitable_trades / len(sell_trades)) if sell_trades else 0.0

		# Max drawdown
		max_drawdown = float(max(self.drawdown_curve)) if self.drawdown_curve else 0.0

		# Sharpe ratio (simple)
		if len(self.equity_curve) > 1:
    pass
    pass
			returns: list[float] = []
			for i in range(1, len(self.equity_curve)):
    pass
    pass
				prev = self.equity_curve[i - 1]
				curr = self.equity_curve[i]
				if prev > 0:
    pass
    pass
					returns.append((curr - prev) / prev)
			if returns:
    pass
    pass
				avg_return = float(np.mean(returns))
				std_return = float(np.std(returns))
				sharpe_ratio = float(avg_return / std_return) if std_return > 0 else 0.0
			else:
				sharpe_ratio = 0.0
		else:
			sharpe_ratio = 0.0

		# Total return
		total_return = (
			float(self.portfolio_value - self.initial_balance) / float(self.initial_balance)
			if self.initial_balance > 0
			else 0.0
		)

		return {
			"total_trades": total_trades,
			"sell_trades": len(sell_trades),
			"win_rate": win_rate,
			"total_pnl": total_pnl,
			"total_cost": total_cost,
			"total_proceeds": total_proceeds,
			"current_portfolio_value": float(self.portfolio_value),
			"max_drawdown": max_drawdown,
			"sharpe_ratio": sharpe_ratio,
			"total_return": total_return,
			"final_equity": self.equity_curve[-1] if self.equity_curve else float(self.initial_balance),
		}

	@handle_errors(
		exceptions=(Exception,),
		default_return={},
		context="detailed analysis generation",
	)
	async def _generate_detailed_analysis(self) -> dict[str, Any]:
		"""Generate detailed analysis of backtest results."""
		try:
			if self.reporter:
    pass
    pass
				return await self.reporter.generate_detailed_report("backtest_analysis")  # type: ignore[attr-defined]
    except Exception as e:
        pass
    except Exception as e:
        pass
			return {}
		except Exception as e:  # pragma: no cover - safety
			self.logger.error(error(f"Error generating detailed analysis: {e}"))
			return {}

	def get_backtest_results(self) -> dict[str, Any]:
    pass
    pass
		"""Get comprehensive backtest results."""
		return {
			"performance_metrics": self._calculate_performance_metrics(),
			"equity_curve": self.equity_curve,
			"drawdown_curve": self.drawdown_curve,
			"trade_history": self.trade_history,
			"current_positions": self.current_position,
			"final_portfolio_value": float(self.portfolio_value),
		}

	@handle_errors(
		exceptions=(Exception,),
		default_return={},
		context="backtest report generation",
	)
	async def generate_backtest_report(
		self,
		report_type: str = "comprehensive",
		export_formats: list[str] | None = None,
	) -> dict[str, Any]:
		"""Generate comprehensive backtest report."""
		if export_formats is None:
    pass
    pass
			export_formats = ["json", "csv", "html"]

		if self.reporter:
    pass
    pass
			try:
				return await self.reporter.generate_detailed_report(report_type, export_formats)  # type: ignore[attr-defined]
    except Exception as e:
        pass
    except Exception as e:
        pass
			except Exception:
				pass
		# Fallback to basic report
		return await self._generate_basic_backtest_report(report_type, export_formats)

	@handle_errors(
		exceptions=(Exception,),
		default_return={},
		context="basic backtest report generation",
	)
	async def _generate_basic_backtest_report(
		self,
		report_type: str,
		export_formats: list[str],
	) -> dict[str, Any]:
		"""Generate basic backtest report when detailed reporter is not available."""
		# Get backtest results
		results = self.get_backtest_results()
		performance_metrics = results["performance_metrics"]

		report_data: dict[str, Any] = {
			"report_type": f"backtest_{report_type}",
			"generated_at": datetime.now().isoformat(),
			"performance_metrics": performance_metrics,
			"equity_curve": results["equity_curve"],
			"drawdown_curve": results["drawdown_curve"],
			"trade_history": results["trade_history"],
			"current_positions": results["current_positions"],
			"final_portfolio_value": results["final_portfolio_value"],
		}

		# Export reports
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		report_dir = "reports/backtesting"
		os.makedirs(report_dir, exist_ok=True)

		for format_type in export_formats:
    pass
    pass
			if format_type == "json":
    pass
    pass
				filename = f"backtest_report_{timestamp}.json"
				filepath = os.path.join(report_dir, filename)
				with open(filepath, "w", encoding="utf-8") as f:
					json.dump(report_data, f, indent=2, default=str)
				self.logger.info(f"✅ Exported backtest JSON report: {filepath}")

		return report_data

	def stop(self) -> None:
    pass
    pass
		"""Stop backtesting."""
		self.is_running = False
		self.logger.info("✅ Enhanced Backtester stopped")


@handle_errors(
	exceptions=(Exception,),
	default_return=None,
	context="enhanced backtester setup",
)
async def setup_enhanced_backtester(
	config: dict[str, Any] | None = None,
) -> EnhancedBacktester | None:
	"""
	Setup enhanced backtester.

	Args:
		config: Configuration dictionary

	Returns:
		EnhancedBacktester: Configured backtester instance
	"""
	try:
		if config is None:
    pass
    pass
			config = {}

    except Exception as e:
        pass
    except Exception as e:
        pass
		backtester = EnhancedBacktester(config)
		success = await backtester.initialize()

		if success:
    pass
    pass
			return backtester
		return None

	except Exception as e:  # pragma: no cover - safety
		system_logger.exception(error(f"Error setting up enhanced backtester: {e}"))
		return None
