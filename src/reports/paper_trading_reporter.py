# src/reports/paper_trading_reporter.py

"""
Paper Trading Reporter for tracking and analyzing paper trading performance.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
import json
import csv
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


@dataclass
class PaperTrade:
    """Represents a single paper trade."""
    trade_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    status: str = "OPEN"  # "OPEN", "CLOSED", "CANCELLED"
    strategy: str = "UNKNOWN"
    notes: str = ""
    
    def close_trade(self, exit_price: float, exit_time: Optional[datetime] = None) -> None:
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.status = "CLOSED"
        
        if self.side == "BUY":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL (short)
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
        self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a specific time."""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    pnl: float
    pnl_percentage: float
    open_trades: int
    closed_trades: int


class PaperTradingReporter:
    """Comprehensive paper trading performance reporter."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the paper trading reporter."""
        self.config = config or {}
        self.logger = system_logger.getChild("PaperTradingReporter")
        
        # Trading data
        self.trades: List[PaperTrade] = []
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        self.initial_capital: float = self.config.get("initial_capital", 100000.0)
        self.current_cash: float = self.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.peak_value: float = self.initial_capital
        
        # Reporting configuration
        self.report_dir = Path(self.config.get("report_dir", "reports/paper_trading"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Paper Trading Reporter initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="paper trading reporter initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the reporter and load existing data if available."""
        try:
            # Load existing trades if available
            await self._load_existing_data()
            
            # Take initial portfolio snapshot
            await self._take_portfolio_snapshot()
            
            self.logger.info("✅ Paper Trading Reporter initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize Paper Trading Reporter: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data loading"
    )
    async def _load_existing_data(self) -> None:
        """Load existing trading data from storage."""
        trades_file = self.report_dir / "trades.json"
        if trades_file.exists():
            try:
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    self.trades = [PaperTrade(**trade_data) for trade_data in data.get("trades", [])]
                    self.current_cash = data.get("current_cash", self.initial_capital)
                    self.positions = data.get("positions", {})
                    
                self.logger.info(f"Loaded {len(self.trades)} existing trades")
                await self._recalculate_performance()
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing data: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance recalculation"
    )
    async def _recalculate_performance(self) -> None:
        """Recalculate performance metrics from loaded trades."""
        self.total_trades = len([t for t in self.trades if t.status == "CLOSED"])
        self.winning_trades = len([t for t in self.trades if t.status == "CLOSED" and t.pnl > 0])
        self.losing_trades = len([t for t in self.trades if t.status == "CLOSED" and t.pnl < 0])
        self.total_pnl = sum(t.pnl for t in self.trades if t.status == "CLOSED")
        
        # Calculate max drawdown
        await self._calculate_max_drawdown()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="portfolio snapshot"
    )
    async def _take_portfolio_snapshot(self) -> None:
        """Take a snapshot of current portfolio state."""
        total_value = self.current_cash
        for symbol, quantity in self.positions.items():
            # For paper trading, we'll use a placeholder price
            # In real implementation, this would get current market price
            placeholder_price = 100.0  # Placeholder
            total_value += quantity * placeholder_price
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            cash=self.current_cash,
            positions=self.positions.copy(),
            pnl=self.total_pnl,
            pnl_percentage=(self.total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0,
            open_trades=len([t for t in self.trades if t.status == "OPEN"]),
            closed_trades=len([t for t in self.trades if t.status == "CLOSED"])
        )
        
        self.portfolio_snapshots.append(snapshot)
        
        # Update peak value
        if total_value > self.peak_value:
            self.peak_value = total_value

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="max drawdown calculation"
    )
    async def _calculate_max_drawdown(self) -> None:
        """Calculate maximum drawdown from portfolio snapshots."""
        if not self.portfolio_snapshots:
            return
            
        peak = self.portfolio_snapshots[0].total_value
        max_dd = 0.0
        
        for snapshot in self.portfolio_snapshots:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            else:
                drawdown = (peak - snapshot.total_value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
        
        self.max_drawdown = max_dd

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="trade execution"
    )
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str = "UNKNOWN",
        notes: str = ""
    ) -> bool:
        """Execute a paper trade."""
        try:
            # Validate trade
            if side not in ["BUY", "SELL"]:
                self.logger.error(f"Invalid trade side: {side}")
                return False
                
            if quantity <= 0:
                self.logger.error(f"Invalid quantity: {quantity}")
                return False
                
            if price <= 0:
                self.logger.error(f"Invalid price: {price}")
                return False
            
            # Check if we have enough cash for buy trades
            if side == "BUY":
                required_cash = quantity * price
                if required_cash > self.current_cash:
                    self.logger.error(f"Insufficient cash. Required: {required_cash}, Available: {self.current_cash}")
                    return False
            
            # Check if we have enough position for sell trades
            if side == "SELL":
                current_position = self.positions.get(symbol, 0)
                if current_position < quantity:
                    self.logger.error(f"Insufficient position. Required: {quantity}, Available: {current_position}")
                    return False
            
            # Create trade
            trade = PaperTrade(
                trade_id=f"PT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.trades)}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                strategy=strategy,
                notes=notes
            )
            
            # Execute trade
            if side == "BUY":
                self.current_cash -= quantity * price
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            else:  # SELL
                self.current_cash += quantity * price
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
            
            # Add trade to list
            self.trades.append(trade)
            self.total_trades += 1
            
            # Take portfolio snapshot
            await self._take_portfolio_snapshot()
            
            self.logger.info(
                f"✅ Executed {side} trade: {quantity} {symbol} @ {price:.2f} "
                f"(Trade ID: {trade.trade_id})"
            )
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to execute trade: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="trade closure"
    )
    async def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None
    ) -> bool:
        """Close an open trade."""
        try:
            trade = next((t for t in self.trades if t.trade_id == trade_id), None)
            if not trade:
                self.logger.error(f"Trade not found: {trade_id}")
                return False
                
            if trade.status != "OPEN":
                self.logger.error(f"Trade {trade_id} is not open (status: {trade.status})")
                return False
            
            # Close the trade
            trade.close_trade(exit_price, exit_time)
            
            # Update performance metrics
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                
            self.total_pnl += trade.pnl
            
            # Take portfolio snapshot
            await self._take_portfolio_snapshot()
            
            self.logger.info(
                f"✅ Closed trade {trade_id}: P&L = {trade.pnl:.2f} "
                f"({trade.pnl_percentage:.2f}%)"
            )
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to close trade: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance report generation"
    )
    async def generate_performance_report(self, format: str = "json") -> Optional[str]:
        """Generate comprehensive performance report."""
        try:
            # Calculate additional metrics
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
            avg_win = sum(t.pnl for t in self.trades if t.status == "CLOSED" and t.pnl > 0) / max(self.winning_trades, 1)
            avg_loss = sum(t.pnl for t in self.trades if t.status == "CLOSED" and t.pnl < 0) / max(self.losing_trades, 1)
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Get current portfolio value
            current_value = self.current_cash
            for symbol, quantity in self.positions.items():
                placeholder_price = 100.0  # Placeholder
                current_value += quantity * placeholder_price
            
            report_data = {
                "summary": {
                    "initial_capital": self.initial_capital,
                    "current_value": current_value,
                    "total_return": current_value - self.initial_capital,
                    "total_return_pct": ((current_value - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0.0,
                    "max_drawdown": self.max_drawdown,
                    "max_drawdown_pct": self.max_drawdown * 100
                },
                "trading_performance": {
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "win_rate": win_rate,
                    "total_pnl": self.total_pnl,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor
                },
                "portfolio": {
                    "cash": self.current_cash,
                    "positions": self.positions,
                    "open_trades": len([t for t in self.trades if t.status == "OPEN"]),
                    "closed_trades": len([t for t in self.trades if t.status == "CLOSED"])
                },
                "generated_at": datetime.now().isoformat()
            }
            
            if format.lower() == "json":
                return json.dumps(report_data, indent=2, default=str)
            elif format.lower() == "csv":
                return await self._generate_csv_report(report_data)
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None
                
        except Exception as e:
            self.logger.exception(f"❌ Failed to generate performance report: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="CSV report generation"
    )
    async def _generate_csv_report(self, report_data: Dict[str, Any]) -> str:
        """Generate CSV format performance report."""
        try:
            csv_file = self.report_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write summary
                writer.writerow(["SUMMARY"])
                writer.writerow(["Metric", "Value"])
                for key, value in report_data["summary"].items():
                    writer.writerow([key, value])
                
                writer.writerow([])
                
                # Write trading performance
                writer.writerow(["TRADING PERFORMANCE"])
                writer.writerow(["Metric", "Value"])
                for key, value in report_data["trading_performance"].items():
                    writer.writerow([key, value])
                
                writer.writerow([])
                
                # Write portfolio
                writer.writerow(["PORTFOLIO"])
                writer.writerow(["Metric", "Value"])
                for key, value in report_data["portfolio"].items():
                    if key == "positions":
                        for symbol, quantity in value.items():
                            writer.writerow([f"position_{symbol}", quantity])
                    else:
                        writer.writerow([key, value])
            
            self.logger.info(f"CSV report generated: {csv_file}")
            return str(csv_file)
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to generate CSV report: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data persistence"
    )
    async def save_data(self) -> bool:
        """Save current trading data to storage."""
        try:
            data = {
                "trades": [
                    {
                        "trade_id": t.trade_id,
                        "symbol": t.symbol,
                        "side": t.side,
                        "quantity": t.quantity,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                        "pnl": t.pnl,
                        "pnl_percentage": t.pnl_percentage,
                        "status": t.status,
                        "strategy": t.strategy,
                        "notes": t.notes
                    }
                    for t in self.trades
                ],
                "current_cash": self.current_cash,
                "positions": self.positions,
                "performance": {
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "total_pnl": self.total_pnl,
                    "max_drawdown": self.max_drawdown,
                    "peak_value": self.peak_value
                }
            }
            
            trades_file = self.report_dir / "trades.json"
            with open(trades_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Trading data saved to {trades_file}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to save trading data: {e}")
            return False

    def get_current_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        current_value = self.current_cash
        for symbol, quantity in self.positions.items():
            placeholder_price = 100.0  # Placeholder
            current_value += quantity * placeholder_price
        
        return {
            "current_value": current_value,
            "cash": self.current_cash,
            "positions": self.positions,
            "open_trades": len([t for t in self.trades if t.status == "OPEN"]),
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown
        }