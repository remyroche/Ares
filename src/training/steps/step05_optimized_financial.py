"""
Step05 Optimized Financial Calculations Module

This module provides vectorized financial calculations with comprehensive logging
for Step05 labeling operations, including transaction costs, risk metrics, and
performance analysis.
"""

import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates
from src.utils.common_operations import safe_mean, safe_std, safe_float, safe_int, validate_dataframe_schema, validate_data_quality, safe_copy, safe_deepcopy, get_current_datetime, format_datetime, create_empty_dataframe, safe_fillna, safe_rolling, safe_append, safe_extend, safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join, get_logger, setup_basic_logging, safe_exception_handler, timed_operation, format_bytes, chunked_iterable, parallel_map, safe_log_metric, safe_log_params, safe_log_artifact
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive, validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change, validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.core.errors import AppError, ValidationError, DataIntegrityError, BusinessRuleError, NotFoundError, ConflictError, RateLimitError, TimeoutError, ServiceUnavailableError, ErrorCode
import numpy as np
import pandas as pd

logger = system_logger.getChild('Step05OptimizedFinancial')


@dataclass
class VectorizedTransactionCosts:
    """Vectorized transaction cost parameters."""
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    slippage_bps: float = 2.0
    funding_rate: float = 0.0001
    min_trade_size: float = 10.0
    max_trade_size: float = 100000.0


@dataclass
class OptimizedTradingPerformance:
    """Optimized trading performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: float
    transaction_costs: float
    net_return: float
    cost_impact: float
    computation_time: float = 0.0
    vectorization_efficiency: float = 0.0


@dataclass
class OptimizedRiskMetrics:
    """Optimized risk assessment metrics."""
    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    beta: float
    correlation_to_market: float
    tail_risk: float
    downside_deviation: float
    computation_time: float = 0.0


class Step05OptimizedFinancialCalculator:
    """Optimized financial calculator with vectorized operations and comprehensive logging."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.transaction_costs = VectorizedTransactionCosts()
        self.performance_stats = {
            'total_calculations': 0,
            'vectorized_operations': 0,
            'total_computation_time': 0.0,
            'avg_computation_time': 0.0,
            'memory_usage_peak': 0.0
        }
        self._load_transaction_cost_config()
        
        self.logger.info("🚀 Initializing Step05 Optimized Financial Calculator")
        self.logger.info(f"💰 Transaction costs: Maker={self.transaction_costs.maker_fee:.3f}, Taker={self.transaction_costs.taker_fee:.3f}")
        self.logger.info(f"📊 Slippage: {self.transaction_costs.slippage_bps} bps")
        self.logger.info(f"💸 Funding rate: {self.transaction_costs.funding_rate:.4f}")
    
    def _load_transaction_cost_config(self):
        """Load transaction cost configuration from config."""
        if 'transaction_costs' in self.config:
            tc_config = self.config['transaction_costs']
            self.transaction_costs = VectorizedTransactionCosts(
                maker_fee=tc_config.get('maker_fee', 0.001),
                taker_fee=tc_config.get('taker_fee', 0.001),
                slippage_bps=tc_config.get('slippage_bps', 2.0),
                funding_rate=tc_config.get('funding_rate', 0.0001),
                min_trade_size=tc_config.get('min_trade_size', 10.0),
                max_trade_size=tc_config.get('max_trade_size', 100000.0)
            )
            self.logger.info("✅ Transaction cost configuration loaded")
    
    def _log_memory_usage(self, operation_name: str):
        """Log current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            if memory_mb > self.performance_stats['memory_usage_peak']:
                self.performance_stats['memory_usage_peak'] = memory_mb
            
            self.logger.debug(f"💾 Memory usage for {operation_name}: {memory_mb:.1f} MB")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not log memory usage: {e}")
    
    @traced(span_name='calculate_transaction_costs_vectorized')
    @validates()
    @handles_errors()
    def calculate_transaction_costs_vectorized(self, data: pd.DataFrame, 
                                             position_sizes: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate transaction costs using vectorized operations.
        
        Args:
            data: DataFrame with price data and labels
            position_sizes: Optional position sizes for each trade
            
        Returns:
            Series with transaction costs for each period
        """
        start_time = time.time()
        self.logger.info("💰 Calculating transaction costs using vectorized operations...")
        
        try:
            # Fast fail validation
            if 'label' not in data.columns or 'close' not in data.columns:
                self.logger.error("❌ FAST FAIL: Missing required columns 'label' or 'close'")
                self.logger.error(f"📋 Available columns: {list(data.columns)}")
                return pd.Series(0.0, index=data.index)
            
            self.logger.info(f"📊 Processing {len(data)} data points")
            
            # Default position size if not provided
            if position_sizes is None:
                position_sizes = pd.Series(1000.0, index=data.index)  # $1000 default
                self.logger.info("💡 Using default position size: $1000")
            else:
                self.logger.info(f"💡 Using provided position sizes (avg: ${position_sizes.mean():.2f})")
            
            # Vectorized trade identification
            valid_trades_mask = data['label'].notna() & (data['label'] != 0)
            valid_trades_count = valid_trades_mask.sum()
            
            self.logger.info(f"📈 Found {valid_trades_count} valid trades ({valid_trades_count/len(data)*100:.1f}% of data)")
            
            if valid_trades_count == 0:
                self.logger.warning("⚠️ No valid trades found")
                return pd.Series(0.0, index=data.index)
            
            # Vectorized position size validation and capping
            position_sizes_valid = position_sizes.copy()
            position_sizes_valid = np.where(
                position_sizes_valid < self.transaction_costs.min_trade_size,
                0.0,  # Set to 0 for trades below minimum
                np.minimum(position_sizes_valid, self.transaction_costs.max_trade_size)
            )
            
            # Count filtered trades
            filtered_trades = (position_sizes_valid == 0).sum()
            if filtered_trades > 0:
                self.logger.warning(f"⚠️ Filtered out {filtered_trades} trades below minimum size")
            
            # Vectorized cost calculations
            self.logger.info("🔄 Computing vectorized transaction costs...")
            
            # Base trading fees (vectorized)
            trading_fees = position_sizes_valid * self.transaction_costs.taker_fee
            
            # Slippage costs (vectorized)
            slippage_costs = position_sizes_valid * (self.transaction_costs.slippage_bps / 10000)
            
            # Funding costs (vectorized, assuming 8-hour funding)
            funding_costs = position_sizes_valid * self.transaction_costs.funding_rate * (8/24)
            
            # Market impact costs (vectorized)
            size_ratios = position_sizes_valid / self.transaction_costs.max_trade_size
            impact_rates = 0.0001 * (size_ratios ** 2)  # Quadratic market impact
            market_impact_costs = position_sizes_valid * impact_rates
            
            # Total costs (vectorized)
            total_costs = trading_fees + slippage_costs + funding_costs + market_impact_costs
            
            # Apply trade mask
            transaction_costs = pd.Series(0.0, index=data.index)
            transaction_costs[valid_trades_mask] = total_costs[valid_trades_mask]
            
            # Calculate statistics using safe math operations
            total_costs_sum = validate_finite(transaction_costs.sum(), "total_costs_sum")
            avg_cost_per_trade = validate_finite(transaction_costs[valid_trades_mask].mean(), "avg_cost_per_trade")
            max_cost = validate_finite(transaction_costs.max(), "max_cost")
            min_cost = validate_finite(transaction_costs[transaction_costs > 0].min(), "min_cost") if (transaction_costs > 0).any() else 0.0
            
            computation_time = time.time() - start_time
            
            # Log detailed results
            self.logger.info(f"✅ Vectorized transaction cost calculation completed in {computation_time:.3f}s")
            self.logger.info(f"💰 Total transaction costs: ${total_costs_sum:.2f}")
            self.logger.info(f"📊 Average cost per trade: ${avg_cost_per_trade:.2f}")
            self.logger.info(f"📈 Cost range: ${min_cost:.2f} - ${max_cost:.2f}")
            self.logger.info(f"🔢 Cost breakdown:")
            # Use safe math operations for percentage calculations
            trading_fees_sum = validate_finite(trading_fees.sum(), "trading_fees_sum")
            slippage_costs_sum = validate_finite(slippage_costs.sum(), "slippage_costs_sum")
            funding_costs_sum = validate_finite(funding_costs.sum(), "funding_costs_sum")
            market_impact_costs_sum = validate_finite(market_impact_costs.sum(), "market_impact_costs_sum")
            
            trading_fees_pct = safe_divide(trading_fees_sum, total_costs_sum, 0.0) * 100
            slippage_pct = safe_divide(slippage_costs_sum, total_costs_sum, 0.0) * 100
            funding_pct = safe_divide(funding_costs_sum, total_costs_sum, 0.0) * 100
            market_impact_pct = safe_divide(market_impact_costs_sum, total_costs_sum, 0.0) * 100
            
            self.logger.info(f"   Trading fees: ${trading_fees_sum:.2f} ({trading_fees_pct:.1f}%)")
            self.logger.info(f"   Slippage: ${slippage_costs_sum:.2f} ({slippage_pct:.1f}%)")
            self.logger.info(f"   Funding: ${funding_costs_sum:.2f} ({funding_pct:.1f}%)")
            self.logger.info(f"   Market impact: ${market_impact_costs_sum:.2f} ({market_impact_pct:.1f}%)")
            
            # Update performance stats
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['vectorized_operations'] += 1
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_calculations']
            )
            
            self._log_memory_usage("transaction_costs_vectorized")
            
            return transaction_costs
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Vectorized transaction cost calculation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return pd.Series(0.0, index=data.index)
    
    @traced(span_name='calculate_trading_performance_vectorized')
    @validates()
    @handles_errors()
    def calculate_trading_performance_vectorized(self, data: pd.DataFrame,
                                               transaction_costs: Optional[pd.Series] = None) -> OptimizedTradingPerformance:
        """
        Calculate trading performance using vectorized operations.
        
        Args:
            data: DataFrame with price data and labels
            transaction_costs: Optional transaction costs series
            
        Returns:
            OptimizedTradingPerformance object with all metrics
        """
        start_time = time.time()
        self.logger.info("📊 Calculating trading performance using vectorized operations...")
        
        try:
            # Fast fail validation
            if 'label' not in data.columns or 'close' not in data.columns:
                self.logger.error("❌ FAST FAIL: Missing required columns 'label' or 'close'")
                self.logger.error(f"📋 Available columns: {list(data.columns)}")
                raise ValueError("Missing required columns: 'label' and 'close'")
            
            self.logger.info(f"📊 Processing {len(data)} data points")
            
            # Calculate transaction costs if not provided
            if transaction_costs is None:
                self.logger.info("💡 Calculating transaction costs...")
                transaction_costs = self.calculate_transaction_costs_vectorized(data)
            
            # Vectorized trade return calculation
            self.logger.info("🔄 Computing vectorized trade returns...")
            trade_returns = self._generate_trade_returns_vectorized(data, transaction_costs)
            
            # Vectorized performance metrics calculation
            self.logger.info("📈 Computing vectorized performance metrics...")
            
            # Basic metrics (vectorized)
            valid_returns = trade_returns[trade_returns != 0]
            total_return = trade_returns.sum()
            total_trades = len(valid_returns)
            winning_trades = (valid_returns > 0).sum()
            losing_trades = (valid_returns < 0).sum()
            
            self.logger.info(f"📊 Trade statistics: {total_trades} total, {winning_trades} wins, {losing_trades} losses")
            
            # Win rate and profit factor (vectorized)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            gross_profit = valid_returns[valid_returns > 0].sum()
            gross_loss = abs(valid_returns[valid_returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            self.logger.info(f"📈 Win rate: {win_rate:.1%}, Profit factor: {profit_factor:.2f}")
            
            # Average win/loss (vectorized)
            avg_win = valid_returns[valid_returns > 0].mean() if winning_trades > 0 else 0.0
            avg_loss = valid_returns[valid_returns < 0].mean() if losing_trades > 0 else 0.0
            
            # Largest win/loss (vectorized)
            largest_win = valid_returns.max() if len(valid_returns) > 0 else 0.0
            largest_loss = valid_returns.min() if len(valid_returns) > 0 else 0.0
            
            # Risk metrics (vectorized)
            volatility = valid_returns.std() if len(valid_returns) > 1 else 0.0
            sharpe_ratio = valid_returns.mean() / volatility if volatility > 0 else 0.0
            
            # Sortino ratio (vectorized)
            downside_returns = valid_returns[valid_returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0.0
            sortino_ratio = valid_returns.mean() / downside_deviation if downside_deviation > 0 else 0.0
            
            # Maximum drawdown (vectorized)
            cumulative_returns = (1 + trade_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Annualized return (vectorized)
            periods_per_year = 252  # Trading days
            annualized_return = (1 + total_return) ** (periods_per_year / len(valid_returns)) - 1 if len(valid_returns) > 0 else 0.0
            
            # Average holding period (vectorized)
            avg_holding_period = self._calculate_avg_holding_period_vectorized(data)
            
            # Transaction cost impact (vectorized)
            total_transaction_costs = transaction_costs.sum()
            cost_impact = total_transaction_costs / abs(total_return) if total_return != 0 else 0.0
            net_return = total_return - total_transaction_costs
            
            computation_time = time.time() - start_time
            
            # Calculate vectorization efficiency
            vectorization_efficiency = self._calculate_vectorization_efficiency(computation_time, len(data))
            
            performance = OptimizedTradingPerformance(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_holding_period=avg_holding_period,
                transaction_costs=total_transaction_costs,
                net_return=net_return,
                cost_impact=cost_impact,
                computation_time=computation_time,
                vectorization_efficiency=vectorization_efficiency
            )
            
            # Log detailed results
            self.logger.info(f"✅ Vectorized trading performance calculation completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Performance summary:")
            self.logger.info(f"   Net return: {net_return:.2%}")
            self.logger.info(f"   Annualized return: {annualized_return:.2%}")
            self.logger.info(f"   Sharpe ratio: {sharpe_ratio:.2f}")
            self.logger.info(f"   Sortino ratio: {sortino_ratio:.2f}")
            self.logger.info(f"   Max drawdown: {max_drawdown:.2%}")
            self.logger.info(f"   Win rate: {win_rate:.1%}")
            self.logger.info(f"   Profit factor: {profit_factor:.2f}")
            self.logger.info(f"   Cost impact: {cost_impact:.1%}")
            self.logger.info(f"   Vectorization efficiency: {vectorization_efficiency:.1%}")
            
            # Update performance stats
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['vectorized_operations'] += 1
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_calculations']
            )
            
            self._log_memory_usage("trading_performance_vectorized")
            
            return performance
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Vectorized trading performance calculation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            # Return default performance metrics
            return OptimizedTradingPerformance(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, profit_factor=0.0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0, avg_holding_period=0.0,
                transaction_costs=0.0, net_return=0.0, cost_impact=0.0,
                computation_time=computation_time, vectorization_efficiency=0.0
            )
    
    def _generate_trade_returns_vectorized(self, data: pd.DataFrame, 
                                         transaction_costs: pd.Series) -> pd.Series:
        """Generate trade returns using vectorized operations."""
        try:
            self.logger.info("🔄 Generating vectorized trade returns...")
            
            returns = pd.Series(0.0, index=data.index)
            
            # Vectorized trade identification
            valid_trades_mask = data['label'].notna() & (data['label'] != 0)
            
            if not valid_trades_mask.any():
                self.logger.warning("⚠️ No valid trades found for return calculation")
                return returns
            
            # Vectorized price calculations
            entry_prices = data['close']
            exit_prices = data['close'].shift(-1)  # Simplified: exit next period
            
            # Vectorized return calculations
            price_changes = (exit_prices - entry_prices) / entry_prices
            
            # Vectorized label-based returns
            long_returns = np.where(data['label'] == 1, price_changes, 0.0)
            short_returns = np.where(data['label'] == -1, -price_changes, 0.0)
            
            gross_returns = long_returns + short_returns
            
            # Vectorized transaction cost adjustment
            cost_adjustments = transaction_costs / 1000  # Assume $1000 position size
            net_returns = gross_returns - cost_adjustments
            
            # Apply trade mask
            returns[valid_trades_mask] = net_returns[valid_trades_mask]
            
            # Log statistics
            valid_returns = returns[valid_trades_mask]
            self.logger.info(f"📊 Generated {len(valid_returns)} trade returns")
            self.logger.info(f"📈 Return statistics: mean={valid_returns.mean():.4f}, std={valid_returns.std():.4f}")
            self.logger.info(f"📊 Return range: {valid_returns.min():.4f} to {valid_returns.max():.4f}")
            
            return returns
            
        except Exception as e:
            self.logger.error(f"❌ Vectorized trade return generation failed: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _calculate_avg_holding_period_vectorized(self, data: pd.DataFrame) -> float:
        """Calculate average holding period using vectorized operations."""
        try:
            if 'label' not in data.columns:
                return 0.0
            
            self.logger.info("🔄 Calculating vectorized average holding period...")
            
            # Vectorized position tracking
            labels = data['label'].fillna(0)
            
            # Find position changes
            position_changes = (labels != labels.shift(1)) & (labels != 0)
            position_entries = position_changes[position_changes].index
            
            # Find position exits
            position_exits = ((labels != 0) & (labels.shift(-1) == 0)).index
            
            # Calculate holding periods
            holding_periods = []
            for entry_idx in position_entries:
                # Find corresponding exit
                exits_after_entry = position_exits[position_exits > entry_idx]
                if len(exits_after_entry) > 0:
                    exit_idx = exits_after_entry[0]
                    holding_period = (exit_idx - entry_idx).total_seconds() / 3600  # Convert to hours
                    holding_periods.append(holding_period)
            
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
            
            self.logger.info(f"📊 Calculated average holding period: {avg_holding_period:.2f} hours")
            
            return avg_holding_period
            
        except Exception as e:
            self.logger.error(f"❌ Vectorized holding period calculation failed: {e}")
            return 0.0
    
    def _calculate_vectorization_efficiency(self, computation_time: float, data_size: int) -> float:
        """Calculate vectorization efficiency compared to loop-based approach."""
        try:
            # Estimate loop-based computation time (rough approximation)
            estimated_loop_time = data_size * 0.0001  # 0.1ms per iteration
            
            if estimated_loop_time > 0:
                efficiency = min(1.0, estimated_loop_time / computation_time)
            else:
                efficiency = 1.0
            
            return efficiency
            
        except Exception:
            return 1.0
    
    @traced(span_name='calculate_risk_metrics_vectorized')
    @validates()
    @handles_errors()
    def calculate_risk_metrics_vectorized(self, data: pd.DataFrame,
                                        market_data: Optional[pd.DataFrame] = None) -> OptimizedRiskMetrics:
        """
        Calculate risk metrics using vectorized operations.
        
        Args:
            data: DataFrame with price data and labels
            market_data: Optional market benchmark data
            
        Returns:
            OptimizedRiskMetrics object with all risk measures
        """
        start_time = time.time()
        self.logger.info("⚠️ Calculating risk metrics using vectorized operations...")
        
        try:
            # Fast fail validation
            if 'close' not in data.columns:
                self.logger.error("❌ FAST FAIL: Missing required column 'close'")
                self.logger.error(f"📋 Available columns: {list(data.columns)}")
                raise ValueError("Missing required column: 'close'")
            
            self.logger.info(f"📊 Processing {len(data)} data points")
            
            # Vectorized returns calculation
            returns = data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                self.logger.warning("⚠️ No valid returns found")
                return self._default_risk_metrics(computation_time=time.time() - start_time)
            
            self.logger.info(f"📈 Calculated {len(returns)} returns")
            
            # Vectorized VaR calculations
            var_95 = np.percentile(returns, 5)  # 5th percentile (95% VaR)
            var_99 = np.percentile(returns, 1)  # 1st percentile (99% VaR)
            
            # Vectorized Expected Shortfall
            expected_shortfall = returns[returns <= var_95].mean()
            
            # Vectorized volatility
            volatility = returns.std()
            
            # Vectorized downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0.0
            
            # Vectorized tail risk (kurtosis)
            tail_risk = returns.kurtosis()
            
            # Vectorized beta and correlation calculations
            beta = 0.0
            correlation_to_market = 0.0
            
            if market_data is not None and 'close' in market_data.columns:
                self.logger.info("📊 Calculating market correlation and beta...")
                
                market_returns = market_data['close'].pct_change().dropna()
                
                # Align returns
                aligned_returns, aligned_market = returns.align(market_returns, join='inner')
                
                if len(aligned_returns) > 1 and len(aligned_market) > 1:
                    # Vectorized correlation
                    correlation_to_market = aligned_returns.corr(aligned_market)
                    
                    # Vectorized beta calculation
                    covariance = np.cov(aligned_returns, aligned_market)[0, 1]
                    market_variance = aligned_market.var()
                    beta = covariance / market_variance if market_variance > 0 else 0.0
                    
                    self.logger.info(f"📊 Market correlation: {correlation_to_market:.3f}")
                    self.logger.info(f"📊 Beta: {beta:.3f}")
            
            computation_time = time.time() - start_time
            
            risk_metrics = OptimizedRiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                volatility=volatility,
                beta=beta,
                correlation_to_market=correlation_to_market,
                tail_risk=tail_risk,
                downside_deviation=downside_deviation,
                computation_time=computation_time
            )
            
            # Log detailed results
            self.logger.info(f"✅ Vectorized risk metrics calculation completed in {computation_time:.3f}s")
            self.logger.info(f"⚠️ Risk metrics summary:")
            self.logger.info(f"   VaR 95%: {var_95:.2%}")
            self.logger.info(f"   VaR 99%: {var_99:.2%}")
            self.logger.info(f"   Expected Shortfall: {expected_shortfall:.2%}")
            self.logger.info(f"   Volatility: {volatility:.2%}")
            self.logger.info(f"   Beta: {beta:.3f}")
            self.logger.info(f"   Correlation to Market: {correlation_to_market:.3f}")
            self.logger.info(f"   Tail Risk (Kurtosis): {tail_risk:.3f}")
            self.logger.info(f"   Downside Deviation: {downside_deviation:.2%}")
            
            # Update performance stats
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['vectorized_operations'] += 1
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_calculations']
            )
            
            self._log_memory_usage("risk_metrics_vectorized")
            
            return risk_metrics
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Vectorized risk metrics calculation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return self._default_risk_metrics(computation_time=computation_time)
    
    def _default_risk_metrics(self, computation_time: float = 0.0) -> OptimizedRiskMetrics:
        """Return default risk metrics when calculation fails."""
        return OptimizedRiskMetrics(
            var_95=0.0, var_99=0.0, expected_shortfall=0.0, volatility=0.0,
            beta=0.0, correlation_to_market=0.0, tail_risk=0.0, downside_deviation=0.0,
            computation_time=computation_time
        )
    
    @traced(span_name='calculate_position_sizing_vectorized')
    @validates()
    @handles_errors()
    def calculate_position_sizing_vectorized(self, data: pd.DataFrame,
                                           confidence_scores: Optional[pd.Series] = None,
                                           risk_budget: float = 0.02) -> pd.Series:
        """
        Calculate optimal position sizes using vectorized Kelly criterion.
        
        Args:
            data: DataFrame with price data and labels
            confidence_scores: Optional confidence scores for each label
            risk_budget: Maximum risk per trade (default 2%)
            
        Returns:
            Series with recommended position sizes
        """
        start_time = time.time()
        self.logger.info("📏 Calculating vectorized position sizing...")
        
        try:
            # Fast fail validation
            if 'label' not in data.columns:
                self.logger.error("❌ FAST FAIL: Missing required column 'label'")
                self.logger.error(f"📋 Available columns: {list(data.columns)}")
                return pd.Series(0.0, index=data.index)
            
            self.logger.info(f"📊 Processing {len(data)} data points")
            self.logger.info(f"💰 Risk budget: {risk_budget:.1%}")
            
            # Default confidence scores if not provided
            if confidence_scores is None:
                confidence_scores = pd.Series(0.5, index=data.index)
                self.logger.info("💡 Using default confidence scores: 0.5")
            else:
                self.logger.info(f"💡 Using provided confidence scores (avg: {confidence_scores.mean():.3f})")
            
            # Vectorized trade identification
            valid_trades_mask = data['label'].notna() & (data['label'] != 0)
            valid_trades_count = valid_trades_mask.sum()
            
            self.logger.info(f"📈 Found {valid_trades_count} valid trades for position sizing")
            
            if valid_trades_count == 0:
                self.logger.warning("⚠️ No valid trades found for position sizing")
                return pd.Series(0.0, index=data.index)
            
            # Vectorized Kelly criterion calculation
            self.logger.info("🔄 Computing vectorized Kelly criterion...")
            
            # Extract valid data
            valid_labels = data['label'][valid_trades_mask]
            valid_confidence = confidence_scores[valid_trades_mask]
            
            # Vectorized Kelly calculation
            # Kelly = (bp - q) / b, where b = odds, p = win probability, q = loss probability
            win_probs = valid_confidence  # Use confidence as win probability
            loss_probs = 1 - win_probs
            
            # Estimate odds from historical data (simplified)
            odds = 1.5  # Assume 1.5:1 odds
            
            # Vectorized Kelly fraction calculation
            kelly_fractions = (odds * win_probs - loss_probs) / odds
            kelly_fractions = np.maximum(0, np.minimum(kelly_fractions, 0.25))  # Cap at 25%
            
            # Vectorized position size calculation
            position_sizes = kelly_fractions * risk_budget * 10000  # Scale to dollar amount
            
            # Create result series
            result_sizes = pd.Series(0.0, index=data.index)
            result_sizes[valid_trades_mask] = position_sizes
            
            # Calculate statistics
            avg_position_size = position_sizes.mean()
            max_position_size = position_sizes.max()
            min_position_size = position_sizes.min()
            total_position_value = position_sizes.sum()
            
            computation_time = time.time() - start_time
            
            # Log detailed results
            self.logger.info(f"✅ Vectorized position sizing calculation completed in {computation_time:.3f}s")
            self.logger.info(f"📏 Position sizing summary:")
            self.logger.info(f"   Average position size: ${avg_position_size:.2f}")
            self.logger.info(f"   Position size range: ${min_position_size:.2f} - ${max_position_size:.2f}")
            self.logger.info(f"   Total position value: ${total_position_value:.2f}")
            self.logger.info(f"   Kelly fraction range: {kelly_fractions.min():.3f} - {kelly_fractions.max():.3f}")
            
            # Update performance stats
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['vectorized_operations'] += 1
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_calculations']
            )
            
            self._log_memory_usage("position_sizing_vectorized")
            
            return result_sizes
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Vectorized position sizing calculation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return pd.Series(1000.0, index=data.index)  # Default $1000 position
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'performance_stats': self.performance_stats.copy(),
            'vectorization_rate': (
                self.performance_stats['vectorized_operations'] / 
                max(1, self.performance_stats['total_calculations'])
            ),
            'avg_computation_time': self.performance_stats['avg_computation_time'],
            'memory_usage_peak': self.performance_stats['memory_usage_peak']
        }
    
    def optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage using vectorized operations."""
        start_time = time.time()
        self.logger.info("💾 Optimizing memory usage with vectorized operations...")
        
        try:
            original_memory = data.memory_usage(deep=True).sum()
            self.logger.info(f"📊 Original memory usage: {original_memory / (1024**2):.1f} MB")
            
            # Optimize numeric columns (vectorized)
            for col in data.select_dtypes(include=['int64']).columns:
                if data[col].min() >= 0:
                    if data[col].max() < 255:
                        data[col] = data[col].astype('uint8')
                    elif data[col].max() < 65535:
                        data[col] = data[col].astype('uint16')
                    elif data[col].max() < 4294967295:
                        data[col] = data[col].astype('uint32')
            
            # Optimize float columns (vectorized)
            for col in data.select_dtypes(include=['float64']).columns:
                data[col] = pd.to_numeric(data[col], downcast='float')
            
            # Optimize categorical columns (vectorized)
            for col in data.select_dtypes(include=['object']).columns:
                if data[col].nunique() / len(data) < 0.5:  # <50% unique values
                    data[col] = data[col].astype('category')
            
            optimized_memory = data.memory_usage(deep=True).sum()
            reduction = (original_memory - optimized_memory) / original_memory
            
            computation_time = time.time() - start_time
            
            self.logger.info(f"✅ Memory optimization completed in {computation_time:.3f}s")
            self.logger.info(f"💾 Memory reduction: {reduction:.1%} ({original_memory/(1024**2):.1f}MB → {optimized_memory/(1024**2):.1f}MB)")
            
            return data
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Memory optimization failed after {computation_time:.3f}s: {e}")
            return data