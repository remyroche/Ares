"""
Trade Reporting Manager

Unified reporting system for both paper and live trading modes.
Generates CSV reports with daily recaps and per-trade analysis.
"""

import asyncio
import csv
import json
import logging
import calendar
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.utils.tprint import (
    tprint_info, tprint_success, tprint_error, tprint_warning,
    tprint_debug, tprint_data_preview, tprint_data_format
)
from src.utils.tprint import tprint


class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    TRADE = "trade"


@dataclass
class TradeRecord:
    """Individual trade record for CSV export"""
    # Trade identification
    trade_id: str
    timestamp: datetime
    exchange: str
    asset: str
    mode: str  # 'paper' or 'trade'
    
    # Trade details
    entry_datetime: datetime
    exit_datetime: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy', 'sell', 'long', 'short'
    direction: str  # 'long' or 'short'
    leverage: float = 1.0  # Leverage used for the trade
    
    # Performance metrics
    net_gain_loss_pct: Optional[float] = None
    net_gain_loss_absolute: Optional[float] = None
    realized_pnl: Optional[float] = None
    gross_pnl: Optional[float] = None  # PnL before fees
    fees: float = 0.0
    slippage_pct: float = 0.0
    
    # Decision reasons
    analyst_confidence: float = 0.0
    tactician_confidence: float = 0.0
    strategist_confidence: float = 0.0
    ensemble_confidence: float = 0.0
    signal_strength: float = 0.0
    
    # SHAP/Feature importance (top 3)
    top_feature_1: str = ""
    top_feature_1_importance: float = 0.0
    top_feature_2: str = ""
    top_feature_2_importance: float = 0.0
    top_feature_3: str = ""
    top_feature_3_importance: float = 0.0
    
    # Context metrics
    regime_1: str = ""
    regime_1_probability: float = 0.0
    regime_2: str = ""
    regime_2_probability: float = 0.0
    regime_3: str = ""
    regime_3_probability: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0
    trend: str = ""
    
    # Execution quality
    execution_time_ms: float = 0.0
    execution_quality: float = 0.0
    
    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for CSV writing"""
        tprint(f"TradingMode.to_csv_dict: Called")
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'exchange': self.exchange,
            'asset': self.asset,
            'mode': self.mode,
            'entry_datetime': self.entry_datetime.isoformat(),
            'exit_datetime': self.exit_datetime.isoformat() if self.exit_datetime else '',
            'entry_price': f"{self.entry_price:.8f}",
            'exit_price': f"{self.exit_price:.8f}" if self.exit_price else '',
            'quantity': f"{self.quantity:.8f}",
            'side': self.side,
            'direction': self.direction,
            'leverage': f"{self.leverage:.2f}",
            'net_gain_loss_pct': f"{self.net_gain_loss_pct:.4f}" if self.net_gain_loss_pct is not None else '',
            'net_gain_loss_absolute': f"{self.net_gain_loss_absolute:.4f}" if self.net_gain_loss_absolute is not None else '',
            'realized_pnl': f"{self.realized_pnl:.4f}" if self.realized_pnl is not None else '',
            'gross_pnl': f"{self.gross_pnl:.4f}" if self.gross_pnl is not None else '',
            'fees': f"{self.fees:.4f}",
            'slippage_pct': f"{self.slippage_pct:.4f}",
            'analyst_confidence': f"{self.analyst_confidence:.4f}",
            'tactician_confidence': f"{self.tactician_confidence:.4f}",
            'strategist_confidence': f"{self.strategist_confidence:.4f}",
            'ensemble_confidence': f"{self.ensemble_confidence:.4f}",
            'signal_strength': f"{self.signal_strength:.4f}",
            'top_feature_1': self.top_feature_1,
            'top_feature_1_importance': f"{self.top_feature_1_importance:.4f}",
            'top_feature_2': self.top_feature_2,
            'top_feature_2_importance': f"{self.top_feature_2_importance:.4f}",
            'top_feature_3': self.top_feature_3,
            'top_feature_3_importance': f"{self.top_feature_3_importance:.4f}",
            'regime_1': self.regime_1,
            'regime_1_probability': f"{self.regime_1_probability:.4f}",
            'regime_2': self.regime_2,
            'regime_2_probability': f"{self.regime_2_probability:.4f}",
            'regime_3': self.regime_3,
            'regime_3_probability': f"{self.regime_3_probability:.4f}",
            'volume': f"{self.volume:.2f}",
            'volatility': f"{self.volatility:.4f}",
            'trend': self.trend,
            'execution_time_ms': f"{self.execution_time_ms:.2f}",
            'execution_quality': f"{self.execution_quality:.4f}",
        }


@dataclass
class DailyRecap:
    """Daily recap record for CSV export"""
    date: date
    exchange: str
    asset: str
    mode: str
    
    # Trading metrics
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    
    # Statistical metrics
    accuracy: float = 0.0  # Win rate
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk metrics
    risk_reward_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_risk: float = 0.0
    
    # Execution metrics
    total_fees: float = 0.0
    avg_slippage_pct: float = 0.0
    avg_execution_quality: Optional[float] = None
    
    # Decision metrics
    avg_confidence: float = 0.0
    avg_analyst_confidence: float = 0.0
    avg_tactician_confidence: float = 0.0
    
    # Context metrics
    primary_regime: str = ""
    avg_volatility: float = 0.0
    avg_volume: float = 0.0
    
    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for CSV writing"""
        tprint(f"TradingMode.to_csv_dict: Called")
        return {
            'date': self.date.isoformat(),
            'exchange': self.exchange,
            'asset': self.asset,
            'mode': self.mode,
            'total_trades': self.total_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': f"{self.total_pnl:.2f}",
            'total_pnl_pct': f"{self.total_pnl_pct:.4f}",
            'gross_profit': f"{self.gross_profit:.2f}",
            'gross_loss': f"{self.gross_loss:.2f}",
            'net_pnl': f"{self.net_pnl:.2f}",
            'accuracy': f"{self.accuracy:.4f}",
            'profit_factor': f"{self.profit_factor:.4f}",
            'avg_win': f"{self.avg_win:.2f}",
            'avg_loss': f"{self.avg_loss:.2f}",
            'largest_win': f"{self.largest_win:.2f}",
            'largest_loss': f"{self.largest_loss:.2f}",
            'risk_reward_ratio': f"{self.risk_reward_ratio:.4f}",
            'sharpe_ratio': f"{self.sharpe_ratio:.4f}",
            'max_drawdown': f"{self.max_drawdown:.4f}",
            'avg_trade_risk': f"{self.avg_trade_risk:.4f}",
            'total_fees': f"{self.total_fees:.2f}",
            'avg_slippage_pct': f"{self.avg_slippage_pct:.4f}",
            'avg_execution_quality': f"{self.avg_execution_quality:.4f}" if self.avg_execution_quality is not None else '',
            'avg_confidence': f"{self.avg_confidence:.4f}",
            'avg_analyst_confidence': f"{self.avg_analyst_confidence:.4f}",
            'avg_tactician_confidence': f"{self.avg_tactician_confidence:.4f}",
            'primary_regime': self.primary_regime,
            'avg_volatility': f"{self.avg_volatility:.4f}",
            'avg_volume': f"{self.avg_volume:.2f}",
        }


class TradeReportingManager:
    """
    Manages trade reporting for both paper and live trading modes.
    
    Generates:
    1. Daily recaps with comprehensive metrics
    2. Per-trade analysis with decision reasons and context
    
    File structure: trade_monitoring/MODE/EXCHANGE/ASSET/
    """
    
    # Default configuration constants
    DEFAULT_ACCOUNT_SIZE = 10000.0  # Can be overridden via config
    
    def __init__(self, base_directory: str = "trade_monitoring", account_size: float = None):
        """
        Initialize trade reporting manager.
        
        Args:
            base_directory: Base directory for all trade reports
            account_size: Account size for percentage calculations (defaults to DEFAULT_ACCOUNT_SIZE)
        """
        tprint(f"TradingMode.__init__: Called")
        self.base_directory = Path(base_directory)
        self.account_size = account_size or self.DEFAULT_ACCOUNT_SIZE
        
        # In-memory storage for current day's trades
        self.current_trades: Dict[str, List[TradeRecord]] = {}  # Key: "mode::exchange::asset"
        
        # Ensure base directory exists
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
        tprint_info(f"📊 Trade reporting manager initialized: {self.base_directory} (account_size: {self.account_size})")
    
    def _get_report_directory(self, mode: str, exchange: str, asset: str) -> Path:
        """
        Get report directory for specific mode/exchange/asset.
        
        Args:
            mode: Trading mode ('paper' or 'trade')
            exchange: Exchange name
            asset: Asset symbol
            
        Returns:
            Path to report directory
        """
        tprint(f"TradingMode._get_report_directory: Called")
        report_dir = self.base_directory / mode / exchange / asset
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    
    def _get_storage_key(self, mode: str, exchange: str, asset: str) -> str:
        """Get storage key for in-memory trade storage"""
        tprint(f"TradingMode._get_storage_key: Called")
        # Use delimiter that's unlikely to appear in exchange/asset names
        return f"{mode}::{exchange}::{asset}"
    
    async def record_trade(self, trade_record: TradeRecord) -> bool:
        """
        Record a trade for reporting.
        
        Args:
            trade_record: Trade record to save
            
        Returns:
            True if successful
        """
        tprint(f"TradingMode.record_trade: Called")
        try:
            # Add to in-memory storage
            storage_key = self._get_storage_key(
                trade_record.mode,
                trade_record.exchange,
                trade_record.asset
            )
            
            if storage_key not in self.current_trades:
                self.current_trades[storage_key] = []
            
            self.current_trades[storage_key].append(trade_record)
            
            # Write to per-trade CSV immediately
            await self._write_trade_to_csv(trade_record)
            
            tprint_success(
                f"✅ Trade recorded: {trade_record.trade_id} "
                f"({trade_record.mode}/{trade_record.exchange}/{trade_record.asset})"
            )
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to record trade: {e}")
            return False

    def _parse_trade_row(self, row: Dict[str, Any], timestamp: datetime) -> Optional[TradeRecord]:
        """Parse a CSV row back into a TradeRecord instance."""
        def _parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value).strip()
            if text == "":
                return default
            try:
                return float(text)
            except Exception:
                return default

        def _parse_datetime(value: Any) -> Optional[datetime]:
            if not value:
                return None
            if isinstance(value, datetime):
                return value
            text = str(value).strip()
            if not text:
                return None
            try:
                return datetime.fromisoformat(text)
            except Exception:
                return None

        try:
            entry_dt = _parse_datetime(row.get("entry_datetime")) or timestamp
            exit_dt = _parse_datetime(row.get("exit_datetime"))

            return TradeRecord(
                trade_id=row.get("trade_id", ""),
                timestamp=timestamp,
                exchange=row.get("exchange", ""),
                asset=row.get("asset", ""),
                mode=row.get("mode", ""),
                entry_datetime=entry_dt,
                exit_datetime=exit_dt,
                entry_price=float(_parse_float(row.get("entry_price"), 0.0) or 0.0),
                exit_price=_parse_float(row.get("exit_price"), None),
                quantity=float(_parse_float(row.get("quantity"), 0.0) or 0.0),
                side=row.get("side", ""),
                direction=row.get("direction", ""),
                leverage=float(_parse_float(row.get("leverage"), 1.0) or 1.0),
                net_gain_loss_pct=_parse_float(row.get("net_gain_loss_pct"), None),
                net_gain_loss_absolute=_parse_float(row.get("net_gain_loss_absolute"), None),
                realized_pnl=_parse_float(row.get("realized_pnl"), None),
                gross_pnl=_parse_float(row.get("gross_pnl"), None),
                fees=float(_parse_float(row.get("fees"), 0.0) or 0.0),
                slippage_pct=float(_parse_float(row.get("slippage_pct"), 0.0) or 0.0),
                analyst_confidence=float(_parse_float(row.get("analyst_confidence"), 0.0) or 0.0),
                tactician_confidence=float(_parse_float(row.get("tactician_confidence"), 0.0) or 0.0),
                strategist_confidence=float(_parse_float(row.get("strategist_confidence"), 0.0) or 0.0),
                ensemble_confidence=float(_parse_float(row.get("ensemble_confidence"), 0.0) or 0.0),
                signal_strength=float(_parse_float(row.get("signal_strength"), 0.0) or 0.0),
                top_feature_1=row.get("top_feature_1", ""),
                top_feature_1_importance=float(_parse_float(row.get("top_feature_1_importance"), 0.0) or 0.0),
                top_feature_2=row.get("top_feature_2", ""),
                top_feature_2_importance=float(_parse_float(row.get("top_feature_2_importance"), 0.0) or 0.0),
                top_feature_3=row.get("top_feature_3", ""),
                top_feature_3_importance=float(_parse_float(row.get("top_feature_3_importance"), 0.0) or 0.0),
                regime_1=row.get("regime_1", ""),
                regime_1_probability=float(_parse_float(row.get("regime_1_probability"), 0.0) or 0.0),
                regime_2=row.get("regime_2", ""),
                regime_2_probability=float(_parse_float(row.get("regime_2_probability"), 0.0) or 0.0),
                regime_3=row.get("regime_3", ""),
                regime_3_probability=float(_parse_float(row.get("regime_3_probability"), 0.0) or 0.0),
                volume=float(_parse_float(row.get("volume"), 0.0) or 0.0),
                volatility=float(_parse_float(row.get("volatility"), 0.0) or 0.0),
                trend=row.get("trend", ""),
                execution_time_ms=float(_parse_float(row.get("execution_time_ms"), 0.0) or 0.0),
                execution_quality=float(_parse_float(row.get("execution_quality"), 0.0) or 0.0),
            )
        except Exception as e:
            tprint_warning(f"⚠️ Failed to parse trade row for historical recap: {e}")
            return None

    def _load_trades_for_date(
        self,
        mode: str,
        exchange: str,
        asset: str,
        target_date: date
    ) -> List[TradeRecord]:
        """Load trades for a specific date from persisted CSV files."""
        trades: List[TradeRecord] = []
        try:
            report_dir = self._get_report_directory(mode, exchange, asset)
            trade_dt = datetime(target_date.year, target_date.month, target_date.day)
            trades_filename = self._get_trade_period_filename(trade_dt)
            trades_file = report_dir / trades_filename
            if not trades_file.exists():
                return []

            with open(trades_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts_str = row.get("timestamp")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except Exception:
                        continue
                    if ts.date() != target_date:
                        continue
                    record = self._parse_trade_row(row, ts)
                    if record is not None:
                        trades.append(record)
        except Exception as e:
            tprint_warning(
                f"⚠️ Failed to load historical trades for {target_date} ({mode}/{exchange}/{asset}): {e}"
            )
        return trades
    
    def _get_trade_period_filename(self, trade_date: datetime) -> str:
        """
        Generate filename for trade CSV based on 15-day periods.
        
        Files are created for:
        - 1st-15th of each month
        - 16th-end of each month
        
        Args:
            trade_date: Date of the trade
            
        Returns:
            Filename for the trade CSV (e.g., "trades_2025-10-01_to_2025-10-15.csv")
        """
        tprint(f"TradingMode._get_trade_period_filename: Called")
        year = trade_date.year
        month = trade_date.month
        day = trade_date.day
        
        if day <= 15:
            # First period: 1st-15th
            start_date = f"{year:04d}-{month:02d}-01"
            end_date = f"{year:04d}-{month:02d}-15"
        else:
            # Second period: 16th-end of month
            # Calculate last day of month
            last_day = calendar.monthrange(year, month)[1]
            start_date = f"{year:04d}-{month:02d}-16"
            end_date = f"{year:04d}-{month:02d}-{last_day:02d}"
        
        return f"trades_{start_date}_to_{end_date}.csv"
    
    async def _write_trade_to_csv(self, trade_record: TradeRecord):
        """
        Write individual trade to per-trade CSV file.
        
        Creates separate files every 15 days:
        - Files for 1st-15th of each month
        - Files for 16th-end of each month
        """
        tprint(f"TradingMode._write_trade_to_csv: Called")
        try:
            report_dir = self._get_report_directory(
                trade_record.mode,
                trade_record.exchange,
                trade_record.asset
            )
            
            # Get period-based filename
            trades_filename = self._get_trade_period_filename(trade_record.timestamp)
            trades_file = report_dir / trades_filename
            
            # Check if file exists to determine if we need headers
            file_exists = trades_file.exists()
            
            # Write trade to CSV
            with open(trades_file, 'a', newline='') as f:
                fieldnames = list(trade_record.to_csv_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    tprint_info(f"📄 Created new trade CSV file: {trades_file}")
                
                writer.writerow(trade_record.to_csv_dict())
            
            # Preview data being written
            tprint_data_preview(
                trade_record.to_csv_dict(), 
                name=f"Trade Data Written to {trades_filename}",
                max_rows=1
            )
            
            tprint_debug(f"📝 Trade written to CSV: {trades_file}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to write trade to CSV: {e}")
    
    async def generate_daily_recap(
        self,
        mode: str,
        exchange: str,
        asset: str,
        target_date: Optional[date] = None
    ) -> bool:
        """
        Generate daily recap for specific mode/exchange/asset.
        
        Args:
            mode: Trading mode
            exchange: Exchange name
            asset: Asset symbol
            target_date: Date to generate recap for (defaults to today)
            
        Returns:
            True if successful
        """
        tprint(f"TradingMode.generate_daily_recap: Called")
        try:
            recap_date = target_date or date.today()
            
            # Get trades for this date from both in-memory storage and persisted CSVs
            storage_key = self._get_storage_key(mode, exchange, asset)
            in_memory_trades = self.current_trades.get(storage_key, [])

            historical_trades = self._load_trades_for_date(mode, exchange, asset, recap_date)

            # Merge and deduplicate by trade_id
            all_trades: List[TradeRecord] = []
            seen_ids = set()
            for t in list(historical_trades) + list(in_memory_trades):
                if t.trade_id in seen_ids:
                    continue
                seen_ids.add(t.trade_id)
                all_trades.append(t)

            # Filter trades for target date
            daily_trades = [
                t for t in all_trades
                if t.timestamp.date() == recap_date
            ]
            
            if not daily_trades:
                tprint_warning(
                    f"⚠️ No trades found for {recap_date} "
                    f"({mode}/{exchange}/{asset})"
                )
                # Still create a zero-activity recap
                daily_trades = []
            
            # Calculate daily recap metrics
            recap = await self._calculate_daily_recap(
                recap_date, mode, exchange, asset, daily_trades
            )
            
            # Write to daily recap CSV
            await self._write_daily_recap_to_csv(recap)
            
            tprint_success(
                f"✅ Daily recap generated: {recap_date} "
                f"({mode}/{exchange}/{asset}) - {recap.total_trades} trades"
            )
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate daily recap: {e}")
            return False
    
    async def _calculate_daily_recap(
        self,
        recap_date: date,
        mode: str,
        exchange: str,
        asset: str,
        trades: List[TradeRecord]
    ) -> DailyRecap:
        """Calculate daily recap metrics from trades"""
        tprint(f"TradingMode._calculate_daily_recap: Called")
        try:
            recap = DailyRecap(
                date=recap_date,
                exchange=exchange,
                asset=asset,
                mode=mode
            )
            
            if not trades:
                return recap
            
            # Basic counts
            recap.total_trades = len(trades)
            recap.long_trades = len([t for t in trades if t.direction == 'long'])
            recap.short_trades = len([t for t in trades if t.direction == 'short'])
            
            # Get closed trades (with exit prices)
            closed_trades = [t for t in trades if t.exit_price is not None]
            
            if closed_trades:
                # Performance metrics
                pnls = [t.realized_pnl for t in closed_trades if t.realized_pnl is not None]
                
                if pnls:
                    recap.winning_trades = len([p for p in pnls if p > 0])
                    recap.losing_trades = len([p for p in pnls if p < 0])
                    
                    recap.total_pnl = sum(pnls)
                    recap.gross_profit = sum(p for p in pnls if p > 0)
                    recap.gross_loss = abs(sum(p for p in pnls if p < 0))
                    recap.net_pnl = recap.total_pnl
                    
                    # Calculate accuracy (win rate)
                    recap.accuracy = recap.winning_trades / len(pnls) if pnls else 0.0
                    
                    # Profit factor
                    recap.profit_factor = (
                        recap.gross_profit / recap.gross_loss
                        if recap.gross_loss > 0 else 0.0
                    )
                    
                    # Average win/loss
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p < 0]
                    
                    recap.avg_win = sum(wins) / len(wins) if wins else 0.0
                    recap.avg_loss = sum(losses) / len(losses) if losses else 0.0
                    recap.largest_win = max(wins) if wins else 0.0
                    recap.largest_loss = min(losses) if losses else 0.0
                    
                    # Risk-reward ratio
                    recap.risk_reward_ratio = (
                        abs(recap.avg_win / recap.avg_loss)
                        if recap.avg_loss != 0 else 0.0
                    )
                    
                    # Sharpe ratio (simplified)
                    import numpy as np
                    if len(pnls) > 1:
                        recap.sharpe_ratio = (
                            np.mean(pnls) / np.std(pnls)
                            if np.std(pnls) > 0 else 0.0
                        )
                    
                    # Max drawdown
                    cumulative_pnl = np.cumsum(pnls)
                    peak = np.maximum.accumulate(cumulative_pnl)
                    drawdown = (peak - cumulative_pnl) / (peak + 1e-8)
                    recap.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
                    
                    # Total PnL percentage (relative to account size)
                    recap.total_pnl_pct = recap.total_pnl / self.account_size if self.account_size > 0 else 0.0
            
            # Execution metrics
            recap.total_fees = sum(t.fees for t in trades)
            slippages = [t.slippage_pct for t in trades if t.slippage_pct > 0]
            recap.avg_slippage_pct = sum(slippages) / len(slippages) if slippages else 0.0
            
            execution_qualities = [t.execution_quality for t in trades if t.execution_quality > 0]
            # Return None if no execution quality data, to distinguish from poor quality (0.0)
            if execution_qualities:
                recap.avg_execution_quality = sum(execution_qualities) / len(execution_qualities)
            else:
                recap.avg_execution_quality = None  # No data available
            
            # Decision metrics
            recap.avg_confidence = sum(t.ensemble_confidence for t in trades) / len(trades)
            recap.avg_analyst_confidence = sum(t.analyst_confidence for t in trades) / len(trades)
            recap.avg_tactician_confidence = sum(t.tactician_confidence for t in trades) / len(trades)
            
            # Context metrics
            regimes = [t.regime_1 for t in trades if t.regime_1]
            if regimes:
                # Most common regime
                from collections import Counter
                regime_counts = Counter(regimes)
                recap.primary_regime = regime_counts.most_common(1)[0][0]
            
            volatilities = [t.volatility for t in trades if t.volatility > 0]
            recap.avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.0
            
            volumes = [t.volume for t in trades if t.volume > 0]
            recap.avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
            
            # Preview calculated recap
            tprint_info(f"📊 Daily Recap Calculated for {recap_date}:")
            tprint_info(f"   Total Trades: {recap.total_trades}")
            tprint_info(f"   Total PnL: ${recap.total_pnl:.2f}")
            tprint_info(f"   Win Rate: {recap.accuracy:.2%}")
            tprint_info(f"   Profit Factor: {recap.profit_factor:.2f}")
            
            return recap
            
        except Exception as e:
            tprint_error(f"❌ Failed to calculate daily recap: {e}")
            return DailyRecap(
                date=recap_date,
                exchange=exchange,
                asset=asset,
                mode=mode
            )
    
    async def _write_daily_recap_to_csv(self, recap: DailyRecap):
        """Write daily recap to CSV file"""
        tprint(f"TradingMode._write_daily_recap_to_csv: Called")
        try:
            report_dir = self._get_report_directory(
                recap.mode,
                recap.exchange,
                recap.asset
            )
            
            # Daily recap CSV file
            recap_file = report_dir / "daily_recap.csv"
            
            # Check if file exists
            file_exists = recap_file.exists()
            
            # If file exists, read existing records and update/append
            if file_exists:
                # Read existing records
                existing_records = []
                with open(recap_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Skip the record for the same date if it exists
                        if row.get('date') != recap.date.isoformat():
                            existing_records.append(row)
                
                # Write all records back
                with open(recap_file, 'w', newline='') as f:
                    fieldnames = list(recap.to_csv_dict().keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write existing records
                    for record in existing_records:
                        writer.writerow(record)
                    
                    # Write new recap
                    writer.writerow(recap.to_csv_dict())
            else:
                # Create new file
                with open(recap_file, 'w', newline='') as f:
                    fieldnames = list(recap.to_csv_dict().keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(recap.to_csv_dict())
            
            # Preview data being written
            tprint_data_preview(
                recap.to_csv_dict(),
                name=f"Daily Recap Data Written",
                max_rows=1
            )
            
            tprint_debug(f"📝 Daily recap written to CSV: {recap_file}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to write daily recap to CSV: {e}")
    
    async def generate_all_daily_recaps(self, target_date: Optional[date] = None) -> bool:
        """
        Generate daily recaps for all tracked mode/exchange/asset combinations.
        
        Args:
            target_date: Date to generate recaps for (defaults to today)
            
        Returns:
            True if all recaps generated successfully
        """
        tprint(f"TradingMode.generate_all_daily_recaps: Called")
        try:
            recap_date = target_date or date.today()
            success = True
            
            for storage_key in self.current_trades.keys():
                # Parse storage key (format: "mode::exchange::asset")
                parts = storage_key.split('::', 2)
                if len(parts) == 3:
                    mode, exchange, asset = parts
                    
                    result = await self.generate_daily_recap(
                        mode, exchange, asset, recap_date
                    )
                    
                    if not result:
                        success = False
            
            if success:
                tprint_success(f"✅ Generated all daily recaps for {recap_date}")
            else:
                tprint_warning(f"⚠️ Some daily recaps failed for {recap_date}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate all daily recaps: {e}")
            return False
    
    def get_trade_count(self, mode: Optional[str] = None,
                       exchange: Optional[str] = None,
                       asset: Optional[str] = None) -> int:
        """Get count of trades matching criteria"""
        tprint(f"TradingMode.get_trade_count: Called")
        count = 0
        for storage_key, trades in self.current_trades.items():
            parts = storage_key.split('::', 2)
            if len(parts) == 3:
                key_mode, key_exchange, key_asset = parts
                
                if mode and key_mode != mode:
                    continue
                if exchange and key_exchange != exchange:
                    continue
                if asset and key_asset != asset:
                    continue
                
                count += len(trades)
        
        return count


# Global instance
trade_reporting_manager = TradeReportingManager()


def create_trade_record_from_execution(
    trade_id: str,
    exchange: str,
    symbol: str,
    mode: str,
    side: str,
    direction: str,
    entry_price: float,
    quantity: float,
    leverage: float = 1.0,
    exit_price: Optional[float] = None,
    exit_datetime: Optional[datetime] = None,
    fees: float = 0.0,
    slippage_pct: float = 0.0,
    trading_decision: Optional[Dict[str, Any]] = None,
    regime_data: Optional[Dict[str, Any]] = None,
    market_context: Optional[Dict[str, Any]] = None
) -> TradeRecord:
    """
    Create a comprehensive TradeRecord from trade execution data.
    
    Args:
        trade_id: Unique trade identifier
        exchange: Exchange name
        symbol: Trading symbol
        mode: 'paper' or 'trade'
        side: 'buy' or 'sell'
        direction: 'long' or 'short'
        entry_price: Entry price
        quantity: Position size
        leverage: Leverage used (default 1.0)
        exit_price: Exit price (optional for open positions)
        exit_datetime: Exit datetime (optional)
        fees: Total fees paid
        slippage_pct: Slippage percentage
        trading_decision: Trading decision dictionary with confidence, signals, etc.
        regime_data: Regime classification data
        market_context: Additional market context (volume, volatility, etc.)
        
    Returns:
        TradeRecord instance
    """
    tprint(f"TradingMode.create_trade_record_from_execution: Called")
    now = datetime.now()
    
    # Calculate PnL if exit price is available
    gross_pnl = None
    realized_pnl = None
    net_gain_loss_pct = None
    net_gain_loss_absolute = None
    
    if exit_price is not None:
        # Calculate gross PnL (before fees)
        if direction == 'long':
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # short
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Calculate net PnL (after fees)
        realized_pnl = gross_pnl - fees
        
        # Calculate percentage gain/loss
        if entry_price > 0:
            net_gain_loss_pct = (realized_pnl / (entry_price * quantity)) * 100
            net_gain_loss_absolute = realized_pnl
    
    # Extract confidence scores from trading decision
    analyst_confidence = 0.0
    tactician_confidence = 0.0
    ensemble_confidence = 0.0
    signal_strength = 0.0
    
    if trading_decision:
        analyst_confidence = trading_decision.get('analyst_confidence', 0.0)
        tactician_confidence = trading_decision.get('tactician_confidence', 0.0)
        ensemble_confidence = trading_decision.get('confidence', 0.0)
        signal_strength = trading_decision.get('signal_strength', 0.0)
        
        # Alternative keys if not found
        if analyst_confidence == 0.0:
            analyst_signal = trading_decision.get('analyst_signal', {})
            if isinstance(analyst_signal, dict):
                analyst_confidence = analyst_signal.get('confidence', 0.0)
        
        if tactician_confidence == 0.0:
            tactician_signal = trading_decision.get('tactician_signal', {})
            if isinstance(tactician_signal, dict):
                tactician_confidence = tactician_signal.get('confidence', 0.0)
    
    # Extract regime information
    regime_1 = ""
    regime_1_probability = 0.0
    regime_2 = ""
    regime_2_probability = 0.0
    regime_3 = ""
    regime_3_probability = 0.0
    
    if regime_data:
        # Handle different regime data formats
        if 'primary_regime' in regime_data:
            regime_1 = regime_data.get('primary_regime', '')
            regime_1_probability = regime_data.get('confidence', 0.0)
        elif 'regime' in regime_data:
            regime_1 = regime_data.get('regime', '')
            regime_1_probability = regime_data.get('regime_probability', 0.0)
        
        # Extract top regimes from probability distribution if available
        regime_probs = regime_data.get('regime_probabilities', {})
        if regime_probs:
            sorted_regimes = sorted(regime_probs.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_regimes) >= 1:
                regime_1 = sorted_regimes[0][0]
                regime_1_probability = sorted_regimes[0][1]
            if len(sorted_regimes) >= 2:
                regime_2 = sorted_regimes[1][0]
                regime_2_probability = sorted_regimes[1][1]
            if len(sorted_regimes) >= 3:
                regime_3 = sorted_regimes[2][0]
                regime_3_probability = sorted_regimes[2][1]
    
    # Extract market context
    volume = 0.0
    volatility = 0.0
    trend = ""
    
    if market_context:
        volume = market_context.get('volume', 0.0)
        volatility = market_context.get('volatility', 0.0)
        trend = market_context.get('trend', '')
    
    # Extract top features from SHAP/feature importance if available
    top_feature_1 = ""
    top_feature_1_importance = 0.0
    top_feature_2 = ""
    top_feature_2_importance = 0.0
    top_feature_3 = ""
    top_feature_3_importance = 0.0
    
    if trading_decision:
        feature_importance = trading_decision.get('feature_importance', {})
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            if len(sorted_features) >= 1:
                top_feature_1 = sorted_features[0][0]
                top_feature_1_importance = sorted_features[0][1]
            if len(sorted_features) >= 2:
                top_feature_2 = sorted_features[1][0]
                top_feature_2_importance = sorted_features[1][1]
            if len(sorted_features) >= 3:
                top_feature_3 = sorted_features[2][0]
                top_feature_3_importance = sorted_features[2][1]
    
    return TradeRecord(
        trade_id=trade_id,
        timestamp=now,
        exchange=exchange,
        asset=symbol,
        mode=mode,
        entry_datetime=now,
        exit_datetime=exit_datetime,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        side=side,
        direction=direction,
        leverage=leverage,
        net_gain_loss_pct=net_gain_loss_pct,
        net_gain_loss_absolute=net_gain_loss_absolute,
        realized_pnl=realized_pnl,
        gross_pnl=gross_pnl,
        fees=fees,
        slippage_pct=slippage_pct,
        analyst_confidence=analyst_confidence,
        tactician_confidence=tactician_confidence,
        strategist_confidence=0.0,  # Reserved for future use
        ensemble_confidence=ensemble_confidence,
        signal_strength=signal_strength,
        top_feature_1=top_feature_1,
        top_feature_1_importance=top_feature_1_importance,
        top_feature_2=top_feature_2,
        top_feature_2_importance=top_feature_2_importance,
        top_feature_3=top_feature_3,
        top_feature_3_importance=top_feature_3_importance,
        regime_1=regime_1,
        regime_1_probability=regime_1_probability,
        regime_2=regime_2,
        regime_2_probability=regime_2_probability,
        regime_3=regime_3,
        regime_3_probability=regime_3_probability,
        volume=volume,
        volatility=volatility,
        trend=trend,
        execution_time_ms=0.0,  # Can be set by caller
        execution_quality=0.0    # Can be set by caller
    )


# Convenience functions
async def record_trade(trade_record: TradeRecord) -> bool:
    """Record a trade for reporting"""
    tprint(f"TradingMode.record_trade: Called")
    return await trade_reporting_manager.record_trade(trade_record)


async def generate_daily_recap(
    mode: str,
    exchange: str,
    asset: str,
    target_date: Optional[date] = None
) -> bool:
    """Generate daily recap for specific mode/exchange/asset"""
    tprint(f"TradingMode.generate_daily_recap: Called")
    return await trade_reporting_manager.generate_daily_recap(
        mode, exchange, asset, target_date
    )


async def generate_all_daily_recaps(target_date: Optional[date] = None) -> bool:
    """Generate daily recaps for all tracked combinations"""
    tprint(f"TradingMode.generate_all_daily_recaps: Called")
    return await trade_reporting_manager.generate_all_daily_recaps(target_date)
