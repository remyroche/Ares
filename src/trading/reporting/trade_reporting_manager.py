"""
Trade Reporting Manager

Unified reporting system for both paper and live trading modes.
Generates CSV reports with daily recaps and per-trade analysis.
"""

import asyncio
import csv
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import calendar

from src.utils.tprint import (
    tprint_info, tprint_success, tprint_error, tprint_warning,
    tprint_debug, tprint_data_preview, tprint_data_format
)


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
    
    # Performance metrics
    net_gain_loss_pct: Optional[float] = None
    net_gain_loss_absolute: Optional[float] = None
    realized_pnl: Optional[float] = None
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
            'net_gain_loss_pct': f"{self.net_gain_loss_pct:.4f}" if self.net_gain_loss_pct is not None else '',
            'net_gain_loss_absolute': f"{self.net_gain_loss_absolute:.4f}" if self.net_gain_loss_absolute is not None else '',
            'realized_pnl': f"{self.realized_pnl:.4f}" if self.realized_pnl is not None else '',
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
    avg_execution_quality: float = 0.0
    
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
            'avg_execution_quality': f"{self.avg_execution_quality:.4f}",
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
    
    def __init__(self, base_directory: str = "trade_monitoring"):
        """
        Initialize trade reporting manager.
        
        Args:
            base_directory: Base directory for all trade reports
        """
        self.base_directory = Path(base_directory)
        
        # In-memory storage for current day's trades
        self.current_trades: Dict[str, List[TradeRecord]] = {}  # Key: "mode_exchange_asset"
        
        # Ensure base directory exists
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
        tprint_info(f"📊 Trade reporting manager initialized: {self.base_directory}")
    
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
        report_dir = self.base_directory / mode / exchange / asset
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    
    def _get_storage_key(self, mode: str, exchange: str, asset: str) -> str:
        """Get storage key for in-memory trade storage"""
        return f"{mode}_{exchange}_{asset}"
    
    async def record_trade(self, trade_record: TradeRecord) -> bool:
        """
        Record a trade for reporting.
        
        Args:
            trade_record: Trade record to save
            
        Returns:
            True if successful
        """
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
            import calendar
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
        try:
            recap_date = target_date or date.today()
            
            # Get trades for this date
            storage_key = self._get_storage_key(mode, exchange, asset)
            all_trades = self.current_trades.get(storage_key, [])
            
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
                    
                    # Total PnL percentage (relative to initial capital - simplified)
                    recap.total_pnl_pct = recap.total_pnl / 10000.0  # Assuming 10k capital
            
            # Execution metrics
            recap.total_fees = sum(t.fees for t in trades)
            slippages = [t.slippage_pct for t in trades if t.slippage_pct > 0]
            recap.avg_slippage_pct = sum(slippages) / len(slippages) if slippages else 0.0
            
            execution_qualities = [t.execution_quality for t in trades if t.execution_quality > 0]
            recap.avg_execution_quality = (
                sum(execution_qualities) / len(execution_qualities)
                if execution_qualities else 0.0
            )
            
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
        try:
            recap_date = target_date or date.today()
            success = True
            
            for storage_key in self.current_trades.keys():
                # Parse storage key
                parts = storage_key.split('_', 2)
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
        count = 0
        for storage_key, trades in self.current_trades.items():
            parts = storage_key.split('_', 2)
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


# Convenience functions
async def record_trade(trade_record: TradeRecord) -> bool:
    """Record a trade for reporting"""
    return await trade_reporting_manager.record_trade(trade_record)


async def generate_daily_recap(
    mode: str,
    exchange: str,
    asset: str,
    target_date: Optional[date] = None
) -> bool:
    """Generate daily recap for specific mode/exchange/asset"""
    return await trade_reporting_manager.generate_daily_recap(
        mode, exchange, asset, target_date
    )


async def generate_all_daily_recaps(target_date: Optional[date] = None) -> bool:
    """Generate daily recaps for all tracked combinations"""
    return await trade_reporting_manager.generate_all_daily_recaps(target_date)
