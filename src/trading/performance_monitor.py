"""
Performance Monitor

This module tracks performance metrics, generates reports, and monitors trading performance.
Integrates with shared utilities for advanced analytics and reporting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json

# Import shared utilities
from ..utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from ..utils.common_utilities import calculate_data_quality_metrics, safe_convert_dtypes
from ..utils.math_validation import safe_divide, validate_finite, safe_log, safe_sqrt
from ..utils.serialization_utils import JSONSerializer, PickleSerializer
from ..utils.tprint import tprint
from ..utils.data.klines_parquet import KlinesParquetManager
from ..utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from ..utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
from ..utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from ..utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from ..utils.matrix_operations import UnifiedMatrixOperations

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    total_return: float
    daily_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    var_95: float
    var_99: float
    expected_shortfall: float


@dataclass
class PerformanceReport:
    """Performance report data structure"""
    timestamp: datetime
    period: str  # 'daily', 'weekly', 'monthly', 'yearly'
    metrics: PerformanceMetrics
    summary: Dict[str, Any]
    recommendations: List[str]
    alerts: List[str]


class PerformanceMonitor:
    """Advanced performance monitoring system with ML analytics"""
    
    def __init__(self, config: Any, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.performance_reports: List[PerformanceReport] = []
        
        # Monitoring
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize utilities
        self._init_utilities()
        
    def _init_utilities(self):
        """Initialize shared utilities"""
        try:
            # Initialize data utilities
            self.klines_manager = KlinesParquetManager()
            self.matrix_ops = UnifiedMatrixOperations()
            
            # Initialize hardware optimizations if available
            if is_m1_available():
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                tprint("✅ M1 hardware optimizations enabled for performance monitoring")
            else:
                self.m1_gpu_manager = None
                self.m1_memory_optimizer = None
                self.m1_cpu_optimizer = None
                tprint("⚠️ M1 hardware optimizations not available")
                
            # Initialize optimization utilities
            self.bayesian_optimizer = BayesianTPEOptimizer()
            
            tprint("✅ Performance monitor utilities initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing performance utilities: {e}")
            tprint(f"⚠️ Some performance utilities may not be available: {e}")
    
    async def start(self) -> None:
        """Start performance monitoring"""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_performance())
        self.logger.info("Performance monitor started")
        tprint("✅ Performance monitor started")
    
    async def stop(self) -> None:
        """Stop performance monitoring"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitor stopped")
        tprint("✅ Performance monitor stopped")
    
    async def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Add a trade to performance tracking"""
        try:
            # Validate trade data
            required_fields = ['timestamp', 'symbol', 'quantity', 'price', 'pnl']
            for field in required_fields:
                if field not in trade_data:
                    self.logger.warning(f"Missing required field {field} in trade data")
                    return
            
            # Add to trade history
            self.trade_history.append(trade_data)
            
            # Calculate daily return
            pnl = trade_data.get('pnl', 0.0)
            self.daily_returns.append(pnl)
            
            # Keep only recent history
            if len(self.trade_history) > 10000:
                self.trade_history = self.trade_history[-10000:]
            
            if len(self.daily_returns) > 1000:
                self.daily_returns = self.daily_returns[-1000:]
            
            tprint(f"✅ Trade added to performance tracking: {trade_data['symbol']} PnL: {pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding trade to performance tracking: {e}")
    
    async def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trade_history:
                return self._get_default_metrics()
            
            # Calculate basic metrics
            total_return = sum(trade.get('pnl', 0.0) for trade in self.trade_history)
            daily_return = sum(self.daily_returns) if self.daily_returns else 0.0
            
            # Calculate risk metrics
            sharpe_ratio = await self._calculate_sharpe_ratio()
            sortino_ratio = await self._calculate_sortino_ratio()
            calmar_ratio = await self._calculate_calmar_ratio()
            max_drawdown = await self._calculate_max_drawdown()
            max_drawdown_duration = await self._calculate_max_drawdown_duration()
            
            # Calculate trade statistics
            winning_trades = [trade for trade in self.trade_history if trade.get('pnl', 0.0) > 0]
            losing_trades = [trade for trade in self.trade_history if trade.get('pnl', 0.0) < 0]
            
            total_trades = len(self.trade_history)
            winning_trades_count = len(winning_trades)
            losing_trades_count = len(losing_trades)
            
            win_rate = safe_divide(winning_trades_count, total_trades, 0.0)
            
            # Calculate profit factor
            gross_profit = sum(trade.get('pnl', 0.0) for trade in winning_trades)
            gross_loss = abs(sum(trade.get('pnl', 0.0) for trade in losing_trades))
            profit_factor = safe_divide(gross_profit, gross_loss, 0.0)
            
            # Calculate average win/loss
            average_win = safe_divide(gross_profit, winning_trades_count, 0.0)
            average_loss = safe_divide(gross_loss, losing_trades_count, 0.0)
            
            # Calculate largest win/loss
            largest_win = max((trade.get('pnl', 0.0) for trade in winning_trades), default=0.0)
            largest_loss = min((trade.get('pnl', 0.0) for trade in losing_trades), default=0.0)
            
            # Calculate consecutive wins/losses
            consecutive_wins = await self._calculate_consecutive_wins()
            consecutive_losses = await self._calculate_consecutive_losses()
            
            # Calculate recovery factor
            recovery_factor = safe_divide(total_return, max_drawdown, 0.0)
            
            # Calculate VaR and Expected Shortfall
            var_95 = await self._calculate_var_95()
            var_99 = await self._calculate_var_99()
            expected_shortfall = await self._calculate_expected_shortfall()
            
            # Create metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=total_return,
                daily_return=daily_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_trades=total_trades,
                winning_trades=winning_trades_count,
                losing_trades=losing_trades_count,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                recovery_factor=recovery_factor,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall
            )
            
            # Store metrics
            self.performance_metrics = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> PerformanceMetrics:
        """Get default performance metrics"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_return=0.0,
            daily_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_wins=0,
            consecutive_losses=0,
            recovery_factor=0.0,
            var_95=0.0,
            var_99=0.0,
            expected_shortfall=0.0
        )
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            
            return safe_divide(mean_return, std_return, 0.0)
            
        except Exception:
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.trade_history:
                return 0.0
            
            # Calculate cumulative returns
            cumulative_returns = []
            running_total = 0.0
            
            for trade in self.trade_history:
                running_total += trade.get('pnl', 0.0)
                cumulative_returns.append(running_total)
            
            if not cumulative_returns:
                return 0.0
            
            # Calculate drawdown
            peak = cumulative_returns[0]
            max_drawdown = 0.0
            
            for cum_return in cumulative_returns:
                if cum_return > peak:
                    peak = cum_return
                drawdown = peak - cum_return
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception:
            return 0.0
    
    async def _calculate_consecutive_wins(self) -> int:
        """Calculate consecutive wins"""
        try:
            if not self.trade_history:
                return 0
            
            consecutive = 0
            max_consecutive = 0
            
            for trade in reversed(self.trade_history):
                pnl = trade.get('pnl', 0.0)
                if pnl > 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    break
            
            return max_consecutive
            
        except Exception:
            return 0
    
    async def _calculate_consecutive_losses(self) -> int:
        """Calculate consecutive losses"""
        try:
            if not self.trade_history:
                return 0
            
            consecutive = 0
            max_consecutive = 0
            
            for trade in reversed(self.trade_history):
                pnl = trade.get('pnl', 0.0)
                if pnl < 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    break
            
            return max_consecutive
            
        except Exception:
            return 0
    
    async def _calculate_var_95(self) -> float:
        """Calculate Value at Risk 95%"""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns_sorted = sorted(self.daily_returns)
            var_index = int(len(returns_sorted) * 0.05)  # 5th percentile
            return abs(returns_sorted[var_index])
            
        except Exception:
            return 0.0
    
    async def _calculate_var_99(self) -> float:
        """Calculate Value at Risk 99%"""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns_sorted = sorted(self.daily_returns)
            var_index = int(len(returns_sorted) * 0.01)  # 1st percentile
            return abs(returns_sorted[var_index])
            
        except Exception:
            return 0.0
    
    async def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            # Calculate VaR 95%
            var_95 = await self._calculate_var_95()
            
            # Calculate expected shortfall
            tail_returns = [r for r in self.daily_returns if r <= -var_95]
            if not tail_returns:
                return 0.0
            
            return abs(np.mean(tail_returns))
            
        except Exception:
            return 0.0
    
    async def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            mean_return = np.mean(self.daily_returns)
            negative_returns = [r for r in self.daily_returns if r < 0]
            
            if not negative_returns:
                return 0.0
            
            downside_std = np.std(negative_returns)
            return safe_divide(mean_return, downside_std, 0.0)
            
        except Exception:
            return 0.0
    
    async def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            if not self.trade_history:
                return 0.0
            
            total_return = sum(trade.get('pnl', 0.0) for trade in self.trade_history)
            max_drawdown = await self._calculate_max_drawdown()
            
            return safe_divide(total_return, max_drawdown, 0.0)
            
        except Exception:
            return 0.0
    
    async def _calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days"""
        try:
            # Simplified calculation
            return 0  # Would need more sophisticated tracking
        except Exception:
            return 0
    
    async def _monitor_performance(self) -> None:
        """Monitor performance continuously"""
        while self._running:
            try:
                # Update performance metrics
                await self.calculate_performance_metrics()
                
                # Wait before next update
                await asyncio.sleep(3600)  # Update every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in performance monitoring: {e}")
                await asyncio.sleep(3600)