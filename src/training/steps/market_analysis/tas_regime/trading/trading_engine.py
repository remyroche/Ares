"""
Trading Engine for TAS

Production-ready trading engine that integrates with tree architecture search
for regime-aware trading execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from .signal_generator import TradingSignalGenerator, SignalConfig

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading modes for the engine."""
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradingConfig:
    """Configuration for trading engine."""

    # Trading mode
    trading_mode: TradingMode = TradingMode.SIMULATION
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%

    # Position sizing
    max_position_size: float = 0.1  # 10% of capital
    min_position_size: float = 0.01  # 1% of capital
    position_sizing_method: str = "fixed_fractional"  # "fixed_fractional", "kelly", "volatility"

    # Risk management
    max_drawdown: float = 0.15  # 15%
    max_daily_loss: float = 0.05  # 5%
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%

    # Regime-specific settings
    enable_regime_aware_trading: bool = True
    regime_confidence_threshold: float = 0.7
    regime_adaptation_enabled: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_update_frequency: int = 100  # Update every N trades

    # Logging and reporting
    enable_trade_logging: bool = True
    log_trades_to_file: bool = True
    log_file_path: str = "trading_logs.json"

    # Advanced features
    enable_ensemble_trading: bool = True
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    enable_dynamic_sizing: bool = True
    volatility_lookback: int = 20

@dataclass
class TradingResult:
    """Result of trading operations."""

    # Trade information
    trade_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    commission: float
    slippage: float

    # Position information
    position_before: float
    position_after: float
    pnl: float
    cumulative_pnl: float

    # Regime information
    regime_id: Optional[str] = None
    regime_confidence: Optional[float] = None
    regime_architecture: Optional[str] = None

    # Risk metrics
    risk_metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class TradingEngine:
    """
    Production-ready trading engine for TAS.

    Integrates tree architecture search with trading execution,
    providing regime-aware trading capabilities.
    """

    def __init__(
        self,
        config: TradingConfig,
        *,
        signal_generator: Optional[TradingSignalGenerator] = None,
        position_manager: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        performance_monitor: Optional[Any] = None,
    ):
        """Initialize trading engine.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.signal_generator = signal_generator or TradingSignalGenerator(SignalConfig())
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.performance_monitor = performance_monitor

        # Trading state
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.regime_architectures = {}
        self.current_regime = None

        # Performance tracking
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = config.initial_capital

        self.logger.info("✅ Trading Engine initialized")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"🎯 Trading mode: {config.trading_mode.value}")

        if self.config.enable_performance_monitoring and self.performance_monitor is None:
            self.logger.debug(
                "Performance monitoring enabled in config but no monitor supplied – skipping external tracker."
            )
        if self.risk_manager is None:
            self.logger.debug("No risk manager supplied – engine will rely on internal safeguards only.")

    def execute_trade(self,
                     symbol: str,
                     side: OrderSide,
                     quantity: float,
                     price: Optional[float] = None,
                     order_type: OrderType = OrderType.MARKET,
                     regime_info: Optional[Dict[str, Any]] = None) -> TradingResult:
        """
        Execute a trade.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Quantity to trade
            price: Optional limit price
            order_type: Type of order
            regime_info: Optional regime information

        Returns:
            Trading result
        """
        self.logger.info(f"📈 Executing {side.value} order for {quantity} {symbol}")

        try:
            # Risk management check
            if self.risk_manager and hasattr(self.risk_manager, "check_trade_risk"):
                is_valid = self.risk_manager.check_trade_risk(
                    symbol, side, quantity, self.current_capital
                )
                if not is_valid:
                    self.logger.warning("⚠️ Trade rejected by risk management")
                    return None

            # Get current price if not provided
            if price is None:
                price = self._get_current_price(symbol)

            # Calculate commission and slippage
            commission = quantity * price * self.config.commission_rate
            slippage = quantity * price * self.config.slippage_rate

            # Adjust price for slippage
            if side == OrderSide.BUY:
                execution_price = price + (slippage / quantity)
            else:
                execution_price = price - (slippage / quantity)

            # Calculate trade value
            trade_value = quantity * execution_price

            # Check capital sufficiency
            if side == OrderSide.BUY and trade_value > self.current_capital:
                self.logger.warning(f"⚠️ Insufficient capital for trade")
                return None

            # Update position
            current_position = self.positions.get(symbol, 0.0)
            new_position = current_position + (quantity if side == OrderSide.BUY else -quantity)

            # Calculate PnL
            pnl = self._calculate_trade_pnl(symbol, side, quantity, execution_price, current_position)

            # Update capital
            if side == OrderSide.BUY:
                self.current_capital -= trade_value + commission
            else:
                self.current_capital += trade_value - commission

            # Update daily PnL
            self.daily_pnl += pnl

            # Update positions
            self.positions[symbol] = new_position
            if self.risk_manager and hasattr(self.risk_manager, "update_position"):
                self.risk_manager.update_position(symbol, new_position)

            # Create trade result
            trade_result = TradingResult(
                trade_id=f"trade_{len(self.trades) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=execution_price,
                commission=commission,
                slippage=slippage,
                position_before=current_position,
                position_after=new_position,
                pnl=pnl,
                cumulative_pnl=self._calculate_cumulative_pnl(),
                regime_id=regime_info.get('regime_id') if regime_info else None,
                regime_confidence=regime_info.get('confidence') if regime_info else None,
                regime_architecture=regime_info.get('architecture') if regime_info else None,
                risk_metrics=self._calculate_risk_metrics(),
                metadata={'config': self.config.__dict__}
            )

            # Store trade
            self.trades.append(trade_result)

            # Update performance monitoring
            if (
                self.config.enable_performance_monitoring
                and self.performance_monitor
                and hasattr(self.performance_monitor, "update_performance")
            ):
                self.performance_monitor.update_performance(trade_result)

            # Log trade
            if self.config.enable_trade_logging:
                self._log_trade(trade_result)

            # Update regime architectures
            if regime_info and 'architecture' in regime_info:
                self.regime_architectures[regime_info['regime_id']] = regime_info['architecture']

            self.logger.info(f"✅ Trade executed: {side.value} {quantity} {symbol} @ ${execution_price:.4f}")
            self.logger.info(f"💰 PnL: ${pnl:.2f}, Capital: ${self.current_capital:,.2f}")

            return trade_result

        except Exception as e:
            self.logger.error(f"❌ Trade execution failed: {e}")
            raise

    def generate_trading_signals(self,
                               market_data: pd.DataFrame,
                               regime_info: Optional[Dict[str, Any]] = None,
                               architecture_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data and regime information.

        Args:
            market_data: Market data (OHLCV)
            regime_info: Regime information
            architecture_info: Tree architecture information

        Returns:
            List of trading signals
        """
        self.logger.info("🎯 Generating trading signals")

        try:
            # Generate signals using signal generator
            signals = self.signal_generator.generate_signals(
                market_data=market_data,
                regime_info=regime_info,
                architecture_info=architecture_info,
                current_positions=self.positions,
                current_capital=self.current_capital
            )

            # Filter signals based on risk management
            filtered_signals = []
            if self.risk_manager and hasattr(self.risk_manager, "check_signal_risk"):
                for signal in signals:
                    if self.risk_manager.check_signal_risk(signal, self.current_capital):
                        filtered_signals.append(signal)
                    else:
                        self.logger.warning(f"⚠️ Signal rejected by risk management: {signal}")
            else:
                filtered_signals = signals

            self.logger.info(f"📊 Generated {len(signals)} signals, {len(filtered_signals)} approved")
            return filtered_signals

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            return []

    def execute_signals(self, signals: List[Dict[str, Any]]) -> List[TradingResult]:
        """
        Execute a list of trading signals.

        Args:
            signals: List of trading signals

        Returns:
            List of trading results
        """
        self.logger.info(f"🚀 Executing {len(signals)} trading signals")

        results = []

        for signal in signals:
            try:
                # Extract signal information
                symbol = signal.get('symbol')
                side = OrderSide(signal.get('side', 'buy'))
                quantity = signal.get('quantity', 0)
                price = signal.get('price')
                order_type = OrderType(signal.get('order_type', 'market'))
                regime_info = signal.get('regime_info')

                # Execute trade
                result = self.execute_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    regime_info=regime_info
                )

                if result:
                    results.append(result)

            except Exception as e:
                self.logger.error(f"❌ Signal execution failed: {e}")
                continue

        self.logger.info(f"✅ Executed {len(results)} trades successfully")
        return results

    def update_regime_architecture(self, regime_id: str, architecture: str):
        """Update regime architecture mapping."""
        self.regime_architectures[regime_id] = architecture
        self.logger.info(f"🔄 Updated architecture for regime {regime_id}")

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()

    def get_current_capital(self) -> float:
        """Get current capital."""
        return self.current_capital

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.trades:
            return {}

        # Calculate performance metrics
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital

        # Calculate Sharpe ratio
        if len(self.trades) > 1:
            pnl_series = [trade.pnl for trade in self.trades]
            sharpe_ratio = np.mean(pnl_series) / np.std(pnl_series) if np.std(pnl_series) > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate win rate
        winning_trades = [trade for trade in self.trades if trade.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # Calculate maximum drawdown
        capital_series = [self.config.initial_capital]
        for trade in self.trades:
            capital_series.append(capital_series[-1] + trade.pnl)

        peak = capital_series[0]
        max_dd = 0
        for capital in capital_series:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)

        return {
            'total_pnl': total_pnl,
            'total_return': total_return,
            'current_capital': self.current_capital,
            'n_trades': len(self.trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'daily_pnl': self.daily_pnl,
            'positions': self.positions.copy()
        }

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (simulation)."""
        # In simulation mode, return a random price
        # In production, this would connect to a real data feed
        return np.random.uniform(100, 200)  # Placeholder

    def _calculate_trade_pnl(self,
                           symbol: str,
                           side: OrderSide,
                           quantity: float,
                           price: float,
                           current_position: float) -> float:
        """Calculate PnL for a trade."""
        # Simplified PnL calculation
        # In production, this would be more sophisticated
        if side == OrderSide.BUY:
            return 0  # No immediate PnL for opening position
        else:
            # PnL for closing/shorting position
            return quantity * price * 0.01  # Simplified calculation

    def _calculate_cumulative_pnl(self) -> float:
        """Calculate cumulative PnL."""
        return sum(trade.pnl for trade in self.trades)

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics."""
        if not self.positions:
            return {}

        # Calculate position risk
        total_exposure = sum(abs(pos) for pos in self.positions.values())
        position_risk = total_exposure / self.current_capital if self.current_capital > 0 else 0

        # Calculate concentration risk
        max_position = max(abs(pos) for pos in self.positions.values()) if self.positions else 0
        concentration_risk = max_position / self.current_capital if self.current_capital > 0 else 0

        return {
            'position_risk': position_risk,
            'concentration_risk': concentration_risk,
            'total_exposure': total_exposure
        }

    def _log_trade(self, trade_result: TradingResult):
        """Log trade to file."""
        if not self.config.log_trades_to_file:
            return

        try:
            log_file = Path(self.config.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert trade result to dict
            trade_dict = {
                'trade_id': trade_result.trade_id,
                'timestamp': trade_result.timestamp.isoformat(),
                'symbol': trade_result.symbol,
                'side': trade_result.side.value,
                'quantity': trade_result.quantity,
                'price': trade_result.price,
                'pnl': trade_result.pnl,
                'regime_id': trade_result.regime_id,
                'regime_confidence': trade_result.regime_confidence
            }

            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(trade_dict) + '\n')

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log trade: {e}")

    def reset_trading_state(self):
        """Reset trading state (for backtesting)."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.config.initial_capital

        if self.risk_manager and hasattr(self.risk_manager, "reset"):
            self.risk_manager.reset()

        self.logger.info("🔄 Trading state reset")

    def export_trading_log(self, filepath: str):
        """Export trading log to file."""
        try:
            log_data = {
                'config': self.config.__dict__,
                'trades': [
                    {
                        'trade_id': trade.trade_id,
                        'timestamp': trade.timestamp.isoformat(),
                        'symbol': trade.symbol,
                        'side': trade.side.value,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'pnl': trade.pnl,
                        'regime_id': trade.regime_id,
                        'regime_confidence': trade.regime_confidence
                    }
                    for trade in self.trades
                ],
                'performance': self.get_performance_metrics()
            }

            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)

            self.logger.info(f"📁 Trading log exported to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export trading log: {e}")
