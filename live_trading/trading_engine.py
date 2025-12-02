"""
Trading Engine

Main engine that coordinates all trading operations.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
import logging

import pandas as pd

from .config import TradingConfig, TradingMode
from .order_manager import OrderManager, Order
from .data_streamer import DataStreamer, StreamData
from .risk_manager import RiskManager
from src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult, MarketData
import os
from pathlib import Path
from src.trading.signal_generation import (
    SignalGenerationPipeline,
    SignalGenerationResult,
    setup_signal_generation_pipeline,
)
from src.trading.monitoring import (
    create_regime_monitor,
    get_regime_monitor,
    RegimeMonitor,
)
from src.trading.config.trading_config import TradingConfig as PipelineTradingConfig, TradingMode as PipelineTradingMode
from src.trading.sizing import (
    PositionSizer,
    LeverageManager,
    RiskCalculator,
    setup_sizing_components,
)


class TradingEngine:
    """Main trading engine that coordinates all trading operations"""
    
    def __init__(self, config: TradingConfig, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.order_manager = OrderManager(config, exchange_client)
        self.data_streamer = DataStreamer(config, exchange_client)
        self.risk_manager = RiskManager(config, exchange_client)
        
        # Trading state
        self._running = False
        self._trading_active = False
        self._last_analysis: Dict[str, AnalysisResult] = {}
        self._last_strategy: Dict[str, StrategyResult] = {}

        # Circuit breaker state
        self._circuit_breaker_open = False
        self._consecutive_failures = 0
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_threshold = config.circuit_breaker_threshold if hasattr(config, 'circuit_breaker_threshold') else 5
        self._circuit_breaker_reset_time = config.circuit_breaker_reset_time if hasattr(config, 'circuit_breaker_reset_time') else 300  # 5 minutes

        # Event handlers
        self.trading_handlers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {
            "on_trade_executed": [],
            "on_risk_violation": [],
            "on_data_received": [],
            "on_error": [],
            "on_circuit_breaker_open": [],
            "on_circuit_breaker_reset": []
        }

        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Signal generation pipelines (one per symbol)
        self.signal_pipelines: Dict[str, SignalGenerationPipeline] = {}

        # Sizing components (one set per symbol, using dampened Kelly stack)
        self.position_sizers: Dict[str, PositionSizer] = {}
        self.leverage_managers: Dict[str, LeverageManager] = {}
        self.sizing_risk_calculators: Dict[str, RiskCalculator] = {}

        # Rolling 1m OHLCV buffers per symbol (for resampling to 15m)
        self._ohlcv_buffers: Dict[str, pd.DataFrame] = {}

        # Rolling standard 15m OHLCV history per symbol (for feature generation)
        self._ohlcv_history_15m: Dict[str, pd.DataFrame] = {}

        # Regime monitoring
        self.regime_monitor: Optional[RegimeMonitor] = None
        
    async def start(self) -> None:
        """Start the trading engine"""
        if self._running:
            return
            
        self.logger.info("Starting trading engine...")
        
        try:
            # Initialize exchange client
            if hasattr(self.exchange_client, '_initialize_exchange'):
                await self.exchange_client._initialize_exchange()
            
            # Load warm-up data
            await self._load_warmup_data()

            # Start components
            await self.order_manager.start()
            await self.data_streamer.start()
            await self.risk_manager.start()

            # Initialize signal generation pipelines and regime monitor
            await self._initialize_signal_pipelines()
            
            # Initialize sizing components (Kelly-based position sizing stack)
            await self._initialize_sizing_components()
            
            # Register event handlers
            self._register_internal_handlers()
            
            # Start trading
            self._running = True
            self._trading_active = True
            
            self.logger.info("Trading engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading engine"""
        if not self._running:
            return
            
        self.logger.info("Stopping trading engine...")
        
        self._running = False
        self._trading_active = False
        
        # Stop components
        await self.order_manager.stop()
        await self.data_streamer.stop()
        await self.risk_manager.stop()
        
        # Close exchange connection
        if hasattr(self.exchange_client, 'close'):
            await self.exchange_client.close()
        
        self.logger.info("Trading engine stopped")

    async def _initialize_signal_pipelines(self) -> None:
        """Initialize signal generation pipelines and regime monitor for live trading.

        One pipeline is created per symbol using the new SignalGenerationPipeline
        that is fully aligned with training artifacts.
        """
        try:
            # Initialize RegimeMonitor once (shared across symbols)
            if self.regime_monitor is None:
                monitor_config: Dict[str, Any] = {
                    "regime_config": {},  # Use default RegimeConfig
                    "stability_threshold": 0.7,
                    "transition_threshold": 0.3,
                    "min_regime_duration_minutes": 30,
                }
                try:
                    self.regime_monitor = await create_regime_monitor(monitor_config)
                    self.logger.info("✅ RegimeMonitor initialized for live trading")
                except Exception as exc:
                    self.logger.warning(f"⚠️ Failed to initialize RegimeMonitor: {exc}")
                    self.regime_monitor = None

            # Create a dedicated signal pipeline per symbol
            for symbol in self.config.symbols:
                if symbol in self.signal_pipelines:
                    continue

                # Build pipeline trading config aligned with training-time config
                pipeline_cfg = PipelineTradingConfig()
                # Attach runtime attributes used by SignalGenerationPipeline
                pipeline_cfg.symbol = symbol
                pipeline_cfg.exchange = getattr(self.config, "exchange_name", "binance")
                pipeline_cfg.direction = "long"
                pipeline_cfg.timeframe = "15m"
                pipeline_cfg.analyst_timeframe = "15m"
                pipeline_cfg.tactician_timeframe = "5m"
                pipeline_cfg.regime_timeframe = "15m"

                pipeline = await setup_signal_generation_pipeline(pipeline_cfg)
                if pipeline is None:
                    self.logger.error(f"❌ Failed to initialize SignalGenerationPipeline for {symbol}")
                    continue

                self.signal_pipelines[symbol] = pipeline
                self.logger.info(f"✅ SignalGenerationPipeline initialized for {symbol}")

        except Exception as exc:
            self.logger.error(f"❌ Failed to initialize signal pipelines: {exc}")
    
    async def _initialize_sizing_components(self) -> None:
        """Initialize Kelly-based sizing components for each symbol.

        This sets up PositionSizer, LeverageManager, and RiskCalculator
        using the same training-aligned TradingConfig used by the
        SignalGenerationPipeline, with optimized parameters applied.
        """
        try:
            for symbol in self.config.symbols:
                # Skip if already initialized
                if symbol in self.position_sizers:
                    continue

                pipeline_cfg = PipelineTradingConfig()
                pipeline_cfg.symbol = symbol
                pipeline_cfg.exchange = getattr(self.config, "exchange_name", "binance")
                pipeline_cfg.direction = "long"
                pipeline_cfg.timeframe = "15m"
                pipeline_cfg.analyst_timeframe = "15m"
                pipeline_cfg.tactician_timeframe = "5m"
                pipeline_cfg.regime_timeframe = "15m"

                try:
                    sizing_components = await setup_sizing_components(pipeline_cfg)
                except Exception as exc:
                    self.logger.warning(f"⚠️ Failed to setup sizing components for {symbol}: {exc}")
                    continue

                position_sizer = sizing_components.get("position_sizer")
                leverage_manager = sizing_components.get("leverage_manager")
                risk_calculator = sizing_components.get("risk_calculator")

                if position_sizer is None:
                    self.logger.warning(
                        f"⚠️ PositionSizer not initialized for {symbol} - falling back to inline sizing"
                    )
                else:
                    self.position_sizers[symbol] = position_sizer

                if leverage_manager is not None:
                    self.leverage_managers[symbol] = leverage_manager
                if risk_calculator is not None:
                    self.sizing_risk_calculators[symbol] = risk_calculator

            self.logger.info("✅ Sizing components initialization completed")

        except Exception as exc:
            self.logger.error(f"❌ Failed to initialize sizing components: {exc}")
    
    def register_handler(self, event_type: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """Register event handler"""
        if event_type in self.trading_handlers:
            self.trading_handlers[event_type].append(handler)
    
    async def execute_trade_decision(self, decision: TradeDecision) -> Optional[Order]:
        """Execute a trade decision"""
        if not self._trading_active:
            self.logger.warning("Trading is not active, ignoring trade decision")
            return None

        # Check circuit breaker
        if await self._check_circuit_breaker():
            self.logger.warning("❌ Circuit breaker is open, rejecting trade decision")
            return None

        try:
            # Validate trade decision with risk manager
            try:
                is_valid, message = await self.risk_manager.validate_trade_decision(decision)

                if not is_valid:
                    self.logger.error(f"❌ Trade decision rejected by risk manager: {message}")
                    await self._notify_handlers("on_risk_violation", {
                        "type": "trade_rejected",
                        "message": message,
                        "decision": decision,
                        "symbol": decision.symbol,
                        "action": decision.action,
                        "quantity": decision.quantity
                    })
                    # Increment failure count
                    await self._record_trade_failure()
                    return None
            except Exception as e:
                self.logger.error(f"❌ Risk validation failed: {e}")
                # Don't proceed if risk validation fails - this is critical
                await self._record_trade_failure()
                return None

            # Create order from decision
            order = await self.order_manager.create_order_from_decision(decision)
            
            # Track trade
            trade_record = {
                "timestamp": datetime.now(),
                "decision": decision,
                "order": order,
                "symbol": decision.symbol,
                "action": decision.action,
                "quantity": decision.quantity,
                "price": decision.price,
                "confidence": decision.confidence,
                "risk_score": decision.risk_score
            }
            self.trade_history.append(trade_record)

            # Record successful trade (reset failure counter)
            await self._record_trade_success()

            # Notify handlers
            await self._notify_handlers("on_trade_executed", trade_record)

            self.logger.info(f"Trade executed: {decision.symbol} {decision.action} {decision.quantity}")

            return order

        except Exception as e:
            self.logger.error(f"Error executing trade decision: {e}")
            # Record trade failure (increment failure counter)
            await self._record_trade_failure()

            await self._notify_handlers("on_error", {
                "type": "trade_execution_error",
                "error": str(e),
                "decision": decision
            })
            return None
    
    async def update_analysis(self, symbol: str, analysis: AnalysisResult) -> None:
        """Update analysis result for a symbol"""
        self._last_analysis[symbol] = analysis
        self.logger.debug(f"Analysis updated for {symbol}: {analysis.signal}")
    
    async def update_strategy(self, symbol: str, strategy: StrategyResult) -> None:
        """Update strategy result for a symbol"""
        self._last_strategy[symbol] = strategy
        self.logger.debug(f"Strategy updated for {symbol}: {strategy.position_bias}")
    
    async def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        order_status = await self.order_manager.get_performance_metrics()
        streaming_status = await self.data_streamer.get_streaming_status()
        risk_summary = await self.risk_manager.get_risk_summary()
        
        # Aggregate signal pipeline metrics per symbol (if available)
        signal_status: Dict[str, Any] = {}
        for symbol, pipeline in self.signal_pipelines.items():
            try:
                signal_status[symbol] = pipeline.get_performance_metrics()
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to get signal metrics for {symbol}: {exc}")
                signal_status[symbol] = {"error": str(exc)}
        
        return {
            "running": self._running,
            "trading_active": self._trading_active,
            "mode": self.config.mode.value,
            "symbols": self.config.symbols,
            "order_manager": order_status,
            "data_streamer": streaming_status,
            "risk_manager": risk_summary,
            "signal_generation": signal_status,
            "total_trades": len(self.trade_history),
            "last_update": datetime.now().isoformat()
        }
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary"""
        positions = {}
        
        for symbol in self.config.symbols:
            try:
                # Get current position from exchange
                account_info = await self.exchange_client.get_account_info()
                
                # Get risk metrics
                risk_metrics = await self.risk_manager.calculate_risk_metrics(symbol)
                
                # Get latest market data
                ticker = await self.exchange_client.get_ticker(symbol)
                current_price = float(ticker.get("last", 0)) if ticker else 0.0
                
                positions[symbol] = {
                    "current_position": risk_metrics.current_position,
                    "position_value": risk_metrics.position_value,
                    "current_price": current_price,
                    "daily_pnl": risk_metrics.daily_pnl,
                    "unrealized_pnl": risk_metrics.unrealized_pnl,
                    "leverage": risk_metrics.leverage,
                    "risk_score": risk_metrics.risk_score,
                    "last_analysis": self._last_analysis.get(symbol),
                    "last_strategy": self._last_strategy.get(symbol)
                }
                
            except Exception as e:
                self.logger.error(f"❌ Error getting position for {symbol}: {e}")
                self.logger.warning(f"⚠️ Using default values for {symbol} - position data may be inaccurate")
                positions[symbol] = {
                    "error": str(e),
                    "current_position": 0.0,
                    "position_value": 0.0,
                    "current_price": 0.0,
                    "daily_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "leverage": 0.0,
                    "risk_score": 0.0
                }
        
        return positions
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_trade_size": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        trade_sizes = []
        
        for trade in self.trade_history:
            # Simplified PnL calculation (would need more sophisticated logic)
            pnl = trade.get("pnl", 0.0)
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1
            
            trade_sizes.append(abs(trade["quantity"]))
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        average_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0.0
        
        # Get risk metrics
        risk_summary = await self.risk_manager.get_risk_summary()
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_trade_size": average_trade_size,
            "max_drawdown": risk_summary.get("max_drawdown", 0.0),
            "sharpe_ratio": risk_summary.get("sharpe_ratio", 0.0),
            "daily_pnl": risk_summary.get("total_daily_pnl", 0.0),
            "risk_violations": risk_summary.get("risk_violations_count", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    async def pause_trading(self) -> None:
        """Pause trading operations"""
        self._trading_active = False
        self.logger.info("Trading paused")
    
    async def resume_trading(self) -> None:
        """Resume trading operations"""
        if self._running:
            self._trading_active = True
            self.logger.info("Trading resumed")
        else:
            self.logger.warning("Cannot resume trading - engine is not running")
    
    async def emergency_stop(self) -> None:
        """Emergency stop - cancel all orders and pause trading"""
        self.logger.warning("⚠️  EMERGENCY STOP TRIGGERED!")

        try:
            # Open circuit breaker immediately
            self._circuit_breaker_open = True
            await self._notify_handlers("on_circuit_breaker_open", {
                "reason": "emergency_stop",
                "consecutive_failures": self._consecutive_failures
            })

            # Cancel all active orders
            active_orders = await self.order_manager.get_active_orders()
            cancelled_count = 0
            failed_count = 0

            for order in active_orders:
                try:
                    await self.order_manager.cancel_order(order.id)
                    cancelled_count += 1
                    self.logger.info(f"✅ Cancelled order: {order.id}")
                except Exception as order_error:
                    failed_count += 1
                    self.logger.error(f"❌ Failed to cancel order {order.id}: {order_error}")

            # Pause trading
            await self.pause_trading()

            if failed_count == 0:
                self.logger.info(f"✅ Emergency stop completed - {cancelled_count} orders cancelled, trading paused")
            else:
                self.logger.warning(f"⚠️  Emergency stop completed with errors - {cancelled_count} orders cancelled, {failed_count} failures")

        except Exception as e:
            self.logger.error(f"❌ Critical error during emergency stop: {e}")
            self.logger.warning("⚠️  Emergency stop may be incomplete - manual intervention required")

    async def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should be opened or if it can be reset.

        Returns:
            True if circuit breaker is open (trading should be blocked)
        """
        if not self._circuit_breaker_open:
            return False

        # Check if enough time has passed to reset circuit breaker
        if self._last_failure_time:
            time_since_failure = (datetime.now() - self._last_failure_time).total_seconds()
            if time_since_failure > self._circuit_breaker_reset_time:
                await self._reset_circuit_breaker()
                return False

        return True

    async def _record_trade_failure(self) -> None:
        """Record a trade failure and check if circuit breaker should be triggered"""
        self._consecutive_failures += 1
        self._last_failure_time = datetime.now()

        if self._consecutive_failures >= self._circuit_breaker_threshold and not self._circuit_breaker_open:
            self._circuit_breaker_open = True
            self.logger.error(
                f"❌ CIRCUIT BREAKER OPENED - {self._consecutive_failures} consecutive failures detected"
            )
            await self._notify_handlers("on_circuit_breaker_open", {
                "consecutive_failures": self._consecutive_failures,
                "threshold": self._circuit_breaker_threshold,
                "last_failure_time": self._last_failure_time.isoformat()
            })

    async def _record_trade_success(self) -> None:
        """Record a successful trade and reset failure counter"""
        if self._consecutive_failures > 0:
            self.logger.info(f"✅ Trade successful - resetting failure counter (was {self._consecutive_failures})")
        self._consecutive_failures = 0

    async def _reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker"""
        if self._circuit_breaker_open:
            self._circuit_breaker_open = False
            self._consecutive_failures = 0
            self.logger.info("✅ Circuit breaker reset - trading resumed")
            await self._notify_handlers("on_circuit_breaker_reset", {
                "reset_time": datetime.now().isoformat()
            })

    async def reset_circuit_breaker_manual(self) -> None:
        """Manually reset the circuit breaker (user-triggered)"""
        await self._reset_circuit_breaker()
        self.logger.info("Circuit breaker manually reset by user")

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "circuit_breaker_open": self._circuit_breaker_open,
            "consecutive_failures": self._consecutive_failures,
            "threshold": self._circuit_breaker_threshold,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "reset_time_seconds": self._circuit_breaker_reset_time
        }

    def _register_internal_handlers(self) -> None:
        """Register internal event handlers"""
        # Order manager handlers
        self.order_manager.register_handler("on_order_filled", self._on_order_filled)
        self.order_manager.register_handler("on_order_failed", self._on_order_failed)
        
        # Data streamer handlers
        self.data_streamer.register_handler("ticker", self._on_ticker_data)
        self.data_streamer.register_handler("trade", self._on_trade_data)
        self.data_streamer.register_handler("kline", self._on_kline_data)
        
        # Risk manager handlers (would be implemented if risk manager had events)
    
    async def _on_order_filled(self, order: Order) -> None:
        """Handle order filled event"""
        self.logger.info(f"Order filled: {order.id}")
        
        # Update position in risk manager
        await self.risk_manager.update_position(
            order.symbol,
            order.filled_quantity if order.side.value == "buy" else -order.filled_quantity,
            order.average_price or order.price or 0.0
        )
        
        # Notify handlers
        await self._notify_handlers("on_trade_executed", {
            "type": "order_filled",
            "order": order,
            "timestamp": datetime.now()
        })
    
    async def _on_order_failed(self, order: Order) -> None:
        """Handle order failed event"""
        self.logger.warning(f"Order failed: {order.id} - {order.error_message}")
        
        # Notify handlers
        await self._notify_handlers("on_error", {
            "type": "order_failed",
            "order": order,
            "error": order.error_message,
            "timestamp": datetime.now()
        })
    
    async def _on_ticker_data(self, stream_data: StreamData) -> None:
        """Handle ticker data"""
        self.logger.debug(f"Ticker data received: {stream_data.symbol}")
        
        # Notify handlers
        await self._notify_handlers("on_data_received", {
            "type": "ticker",
            "symbol": stream_data.symbol,
            "data": stream_data.data,
            "timestamp": stream_data.timestamp
        })
    
    async def _on_trade_data(self, stream_data: StreamData) -> None:
        """Handle trade data"""
        self.logger.debug(f"Trade data received: {stream_data.symbol}")
        
        # Notify handlers
        await self._notify_handlers("on_data_received", {
            "type": "trade",
            "symbol": stream_data.symbol,
            "data": stream_data.data,
            "timestamp": stream_data.timestamp
        })
    
    async def _on_kline_data(self, stream_data: StreamData) -> None:
        """Handle kline data"""
        self.logger.debug(f"Kline data received: {stream_data.symbol}")
        symbol = stream_data.symbol

        # Convert to MarketData (1m bar)
        market_data = MarketData(
            symbol=symbol,
            timestamp=stream_data.timestamp,
            open=stream_data.data["open"],
            high=stream_data.data["high"],
            low=stream_data.data["low"],
            close=stream_data.data["close"],
            volume=stream_data.data["volume"],
            interval=stream_data.data["interval"],
        )

        # Maintain rolling 1m OHLCV buffer per symbol
        row = pd.DataFrame(
            {
                "open": [market_data.open],
                "high": [market_data.high],
                "low": [market_data.low],
                "close": [market_data.close],
                "volume": [market_data.volume],
            },
            index=[market_data.timestamp],
        )

        # Save raw 1m data immediately
        await self._save_market_data(symbol, row, "1m")

        buf = self._ohlcv_buffers.get(symbol)
        if buf is None:
            buf = row
        else:
            # Handle duplicates if timestamp already exists
            if market_data.timestamp in buf.index:
                buf = buf.drop(market_data.timestamp)
            buf = pd.concat([buf, row])
            # Keep a large enough history window for 15m resampling and rolling windows
            # 5 days * 24 hours * 60 minutes = 7200 rows. Let's keep 10000.
            buf = buf.sort_index().last("10000min")
        self._ohlcv_buffers[symbol] = buf

        # Notify external handlers with raw kline data (unchanged behavior)
        await self._notify_handlers(
            "on_data_received",
            {
                "type": "kline",
                "symbol": symbol,
                "market_data": market_data,
                "timestamp": stream_data.timestamp,
            },
        )

        # --- 15m Bar Maintenance ---

        # Check if a standard 15m bar has closed (00, 15, 30, 45)
        # Assuming the kline timestamp represents the OPEN time or CLOSE time?
        # StreamData usually has close time or we infer it.
        # If timestamp is 10:14:00 (1m bar), it closes at 10:15:00.
        # Let's assume `market_data.timestamp` is the closing time of the 1m bar.
        # If timestamp.minute is 0, 15, 30, 45, a 15m interval just finished.

        if market_data.timestamp.minute % 15 == 0:
            # Resample strictly up to this timestamp to get the closed 15m bar
            try:
                # Resample whole buffer to standard 15T
                resampled_15m = (
                    buf.sort_index()
                    .resample("15T", label='right', closed='right') # align to close time
                    .agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    })
                    .dropna()
                )

                # Get the last bar which should match current timestamp
                if not resampled_15m.empty:
                    last_bar = resampled_15m.iloc[[-1]]
                    if last_bar.index[-1] == market_data.timestamp:
                        # Update persistent 15m history
                        hist_15m = self._ohlcv_history_15m.get(symbol)
                        if hist_15m is None:
                            hist_15m = last_bar
                        else:
                            # Avoid duplicates
                            if last_bar.index[-1] in hist_15m.index:
                                hist_15m = hist_15m.drop(last_bar.index[-1])
                            hist_15m = pd.concat([hist_15m, last_bar])
                            # Keep ~60 days of 15m data (4 * 24 * 60 = 5760 rows)
                            hist_15m = hist_15m.sort_index().tail(6000)

                        self._ohlcv_history_15m[symbol] = hist_15m

                        # Save the closed 15m bar
                        await self._save_market_data(symbol, last_bar, "15m")
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to process standard 15m bar for {symbol}: {exc}")

        # --- Signal Generation Logic ---

        # If we have a signal pipeline for this symbol and enough 15m history
        pipeline = self.signal_pipelines.get(symbol)

        # Check if we should update signal (every 5 minutes)
        # Trigger on 0, 5, 10, 15... minutes
        if market_data.timestamp.minute % 5 != 0:
            return

        if pipeline is None:
            return

        # Prepare input data for pipeline:
        # 1. Start with standard closed 15m history
        input_data = self._ohlcv_history_15m.get(symbol)
        if input_data is None or len(input_data) < 50: # Need decent history for indicators
            # Try to build from buffer if history is missing (e.g. cold start without warmup)
            try:
                input_data = (
                    buf.sort_index()
                    .resample("15T", label='right', closed='right')
                    .agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    })
                    .dropna()
                )
            except Exception:
                return # Not enough data

        # 2. Construct "Ghost Bar" (Rolling 15m window ending NOW)
        # This captures the latest 15m of market action, even if off-grid.
        # e.g. at 10:05, window is 09:51 - 10:05
        # We need the last 15 minutes of 1m data from buffer
        try:
            last_15_mins_1m = buf.sort_index().last("15min")
            if not last_15_mins_1m.empty:
                ghost_bar = pd.DataFrame({
                    "open": [last_15_mins_1m["open"].iloc[0]],
                    "high": [last_15_mins_1m["high"].max()],
                    "low": [last_15_mins_1m["low"].min()],
                    "close": [last_15_mins_1m["close"].iloc[-1]],
                    "volume": [last_15_mins_1m["volume"].sum()]
                }, index=[market_data.timestamp]) # Use current 1m timestamp as ghost bar timestamp

                # 3. Append Ghost Bar to history (if it's not already the last closed bar)
                # If we are at 10:00 (divisible by 15), the standard logic above might have added it.
                # If prediction triggers at 10:00, standard logic puts 10:00 bar in history.
                # If prediction triggers at 10:05, history has 10:00. We append 10:05.

                # Copy to avoid modifying persistent history
                prediction_input = input_data.copy()

                if ghost_bar.index[-1] not in prediction_input.index:
                    prediction_input = pd.concat([prediction_input, ghost_bar])
                else:
                    # Update existing (if for some reason it exists)
                    prediction_input.loc[ghost_bar.index[-1]] = ghost_bar.iloc[0]
            else:
                # Should not happen if buf is populated
                return
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to construct ghost bar for {symbol}: {exc}")
            return

        # Get current account balance for position sizing (best-effort)
        account_balance: float = 0.0
        try:
            account_info = await self.exchange_client.get_account_info()
            if account_info:
                account_balance = float(
                    account_info.get("totalBalance", account_info.get("balance", 0.0))
                )
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to fetch account balance for sizing: {exc}")

        try:
            result: SignalGenerationResult = await pipeline.generate_signal(
                symbol=symbol,
                market_data=prediction_input,
                additional_features=None,
            )

            # Update RegimeMonitor with latest regime probabilities
            if self.regime_monitor is not None and result and result.hmm_output:
                try:
                    await self.regime_monitor.update_regime_state(
                        regime_probabilities=result.hmm_output.regime_probabilities,
                        confidence=result.hmm_output.confidence,
                        market_conditions={
                            "symbol": symbol,
                            "timestamp": result.timestamp,
                        },
                    )
                except Exception as exc:
                    self.logger.warning(f"⚠️ Failed to update RegimeMonitor: {exc}")

            # Map SignalGenerationResult -> TradeDecision and execute it
            decision = await self._map_signal_to_trade_decision(
                result,
                prediction_input,
                account_balance,
            )
            if decision is not None:
                await self.execute_trade_decision(decision)

        except Exception as exc:
            self.logger.error(f"❌ Failed to generate or execute signal for {symbol}: {exc}")

    async def _save_market_data(self, symbol: str, data: pd.DataFrame, timeframe: str) -> None:
        """Save market data to Parquet files."""
        try:
            exchange = getattr(self.config, "exchange_name", "binance")
            base_dir = Path(f"live_data/{exchange}/{symbol}/{timeframe}")
            base_dir.mkdir(parents=True, exist_ok=True)

            # Partition by date to avoid huge files
            # Assuming data has DatetimeIndex
            for date, group in data.groupby(data.index.date):
                filename = base_dir / f"{date}.parquet"

                # Check if file exists to append or create
                if filename.exists():
                    try:
                        existing = pd.read_parquet(filename)
                        # Combine and drop duplicates
                        combined = pd.concat([existing, group])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        combined = combined.sort_index()
                        combined.to_parquet(filename)
                    except Exception as e:
                        self.logger.error(f"Failed to append to {filename}: {e}")
                        # Fallback: save separate file with timestamp
                        ts_str = datetime.now().strftime("%H%M%S")
                        fallback = base_dir / f"{date}_{ts_str}.parquet"
                        group.to_parquet(fallback)
                else:
                    group.to_parquet(filename)

        except Exception as e:
            self.logger.error(f"❌ Failed to save market data for {symbol}: {e}")

    async def _load_warmup_data(self) -> None:
        """Load warm-up data for indicators."""
        try:
            exchange = getattr(self.config, "exchange_name", "binance")

            for symbol in self.config.symbols:
                # Path pattern: data/historical/{exchange}/{symbol}/1m.parquet
                warmup_path = Path(f"data/historical/{exchange}/{symbol}/1m.parquet")

                if warmup_path.exists():
                    self.logger.info(f"🔄 Loading warm-up data for {symbol} from {warmup_path}")
                    try:
                        df = pd.read_parquet(warmup_path)
                        if not df.empty:
                            # Ensure index is datetime and sorted
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            df = df.sort_index()

                            # Populate 1m buffer
                            # Keep last ~5-7 days for 1m buffer
                            one_week_ago = df.index[-1] - pd.Timedelta(days=7)
                            recent_1m = df[df.index >= one_week_ago]
                            self._ohlcv_buffers[symbol] = recent_1m

                            # Populate 15m history
                            # Resample all available history to 15m
                            hist_15m = (
                                df.resample("15T", label='right', closed='right')
                                .agg({
                                    "open": "first",
                                    "high": "max",
                                    "low": "min",
                                    "close": "last",
                                    "volume": "sum",
                                })
                                .dropna()
                            )
                            # Keep last 6000 bars
                            hist_15m = hist_15m.tail(6000)
                            self._ohlcv_history_15m[symbol] = hist_15m

                            self.logger.info(f"✅ Loaded {len(recent_1m)} 1m bars and {len(hist_15m)} 15m bars for {symbol}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to read warm-up file {warmup_path}: {e}")
                else:
                    self.logger.warning(
                        f"⚠️ Warm-up data not found at {warmup_path}. "
                        "Indicators will need time to converge."
                    )
        except Exception as e:
            self.logger.error(f"❌ Error during warm-up data loading: {e}")

    async def _map_signal_to_trade_decision(
        self,
        result: SignalGenerationResult,
        market_data_15m: pd.DataFrame,
        account_balance: float,
    ) -> Optional[TradeDecision]:
        """Convert SignalGenerationResult into a live TradeDecision.

        This adapter ensures that live trades use the same optimization
        parameters (sizing, leverage, TPSL) as the backtested
        final_parameters_optimization step, and prefers the Kelly-based
        PositionSizer when available.
        """
        try:
            if result is None:
                return None

            action = result.final_signal
            if action not in {"buy", "sell", "hold", "close"}:
                self.logger.warning(f"⚠️ Unknown final signal '{action}', treating as 'hold'")
                return None

            # No trade on hold
            if action == "hold":
                return None

            # Current price from latest 15m bar
            if market_data_15m is None or market_data_15m.empty:
                return None
            current_price = float(market_data_15m["close"].iloc[-1])
            if current_price <= 0:
                return None

            params = result.optimization_parameters or {}

            # Leverage-independent TPSL levels from optimized parameters
            stop_loss_pct = float(params.get("stop_loss_pct", 0.03))
            take_profit_pct = float(params.get("take_profit_pct", 0.06))

            if action in {"buy", "close"}:
                stop_loss = current_price * (1.0 - stop_loss_pct)
                take_profit = current_price * (1.0 + take_profit_pct)
            else:  # sell / short
                stop_loss = current_price * (1.0 + stop_loss_pct)
                take_profit = current_price * (1.0 - take_profit_pct)

            # Base confidence-anchored risk score
            confidence = float(result.final_confidence)
            risk_score = float(max(0.0, min(1.0, 1.0 - confidence)))

            quantity: float
            leverage: float

            # Prefer Kelly-based PositionSizer if available and account_balance is valid
            sizer = self.position_sizers.get(result.symbol)
            if (
                sizer is not None
                and getattr(sizer, "is_initialized", False)
                and account_balance > 0
            ):
                try:
                    ml_predictions: Dict[str, Any] = {
                        "combined_confidence": float(
                            getattr(result.tactician_output, "combined_confidence", confidence)
                        )
                    }

                    analyst_conf = float(
                        getattr(result.analyst_output, "analyst_confidence", confidence)
                    )
                    tactician_conf = float(
                        getattr(result.tactician_output, "tactician_confidence", confidence)
                    )

                    size_result = await sizer.calculate_position_size(
                        symbol=result.symbol,
                        ml_predictions=ml_predictions,
                        current_price=current_price,
                        account_balance=account_balance,
                        analyst_confidence=analyst_conf,
                        tactician_confidence=tactician_conf,
                        stop_loss_price=stop_loss,
                        volatility=None,
                        market_data=None,
                    )

                    notional = float(size_result.recommended_size)
                    quantity = max(notional / current_price, 0.0)
                    leverage = float(size_result.leverage)

                    # Align risk_score with Kelly confidence if available
                    kelly_conf = float(getattr(size_result, "confidence", confidence))
                    risk_score = float(max(0.0, min(1.0, 1.0 - kelly_conf)))
                except Exception as exc:
                    self.logger.warning(
                        f"⚠️ PositionSizer failed for {result.symbol}, falling back to inline sizing: {exc}"
                    )
                    sizer = None  # Trigger fallback path

            if (
                sizer is None
                or not getattr(sizer, "is_initialized", False)
                or account_balance <= 0
            ):
                # Fallback: basic sizing using optimized parameters and confidence
                max_notional = float(self.config.max_position_size)
                sizing_factor = float(params.get("position_sizing_factor", 0.02))
                notional = max_notional * sizing_factor * confidence
                quantity = max(notional / current_price, 0.0)
                leverage = float(params.get("leverage_multiplier", 1.0))

            return TradeDecision(
                timestamp=result.timestamp,
                symbol=result.symbol,
                action=action,
                quantity=quantity,
                price=current_price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                risk_score=risk_score,
            )

        except Exception as exc:
            self.logger.error(f"❌ Failed to map SignalGenerationResult to TradeDecision: {exc}")
            return None
    
    async def _notify_handlers(self, event_type: str, data: Any) -> None:
        """Notify registered handlers"""
        if event_type in self.trading_handlers:
            for handler in self.trading_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"❌ Error in trading handler: {e}")
                    self.logger.warning("⚠️ Handler failed - continuing with other handlers")
    
    async def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        trades = self.trade_history
        
        if symbol:
            trades = [trade for trade in trades if trade["symbol"] == symbol]
        
        return trades[-limit:] if limit > 0 else trades