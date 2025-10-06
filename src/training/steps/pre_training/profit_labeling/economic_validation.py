"""
Economic Validation Layer for Profit Labeling

This module implements economic validation to assess label quality via simulated P&L,
ensuring labels represent tradeable opportunities after accounting for transaction costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import safe_divide, safe_mean, safe_std


class EconomicMetric(Enum):
    """Enumeration of economic metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return"
    VOLATILITY = "volatility"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"


@dataclass
class EconomicConfig:
    """Configuration for economic validation."""

    # Transaction costs
    transaction_cost_bps: float = 8.0  # 0.08% as basis points
    slippage_bps: float = 2.0  # 0.02% slippage estimate
    market_impact_factor: float = 0.1  # Market impact scaling factor

    # Position sizing
    max_position_size: float = 1.0  # Maximum position size as fraction of capital
    position_size_method: str = "equal"  # "equal", "kelly", "volatility_scaled"

    # Risk management
    stop_loss_pct: Optional[float] = None  # Stop loss as percentage
    take_profit_pct: Optional[float] = None  # Take profit as percentage
    max_drawdown_limit: float = 0.3  # 30% maximum drawdown

    # Simulation settings
    initial_capital: float = 100000.0
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"
    include_borrowing_costs: bool = False
    borrowing_rate: float = 0.05  # 5% annual borrowing rate

    # Backtest settings
    warmup_periods: int = 50  # Periods to skip at start for reliable metrics
    min_trades: int = 10  # Minimum trades for reliable evaluation


@dataclass
class EconomicMetrics:
    """Container for economic validation metrics."""

    # Core performance metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Advanced metrics
    calmar_ratio: float = 0.0  # Sharpe ratio / Max drawdown
    sortino_ratio: float = 0.0  # Return / Downside deviation
    information_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Economic utility score
    economic_quality_score: float = 0.0

    # Metadata
    n_periods: int = 0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class EconomicValidator:
    """
    Economic Validator for assessing label quality via simulated P&L.

    This class simulates trading performance to ensure labels represent
    economically viable trading opportunities after accounting for costs.
    """

    def __init__(self, config: Optional[EconomicConfig] = None):
        """Initialize economic validator."""
        self.config = config or EconomicConfig()
        self.logger = logging.getLogger('EconomicValidator')

        tprint_success("💰 Economic Validator initialized")
        tprint_info(f"   → Transaction costs: {self.config.transaction_cost_bps}bps")
        tprint_info(f"   → Initial capital: ${self.config.initial_capital:,.0f}")
        tprint_info(f"   → Position sizing: {self.config.position_size_method}")

    def calculate_pnl_quality(self, labels: pd.DataFrame, prices: pd.DataFrame,
                            confidence_scores: Optional[pd.DataFrame] = None,
                            eligibility_masks: Optional[pd.DataFrame] = None) -> Dict[str, EconomicMetrics]:
        """
        Assess label quality via simulated P&L.

        Args:
            labels: Label DataFrame with target columns
            prices: Price data for P&L calculation
            confidence_scores: Optional confidence scores for position sizing
            eligibility_masks: Optional eligibility masks

        Returns:
            Dictionary mapping target names to EconomicMetrics
        """
        start_time = datetime.now()
        tprint_info("💰 Calculating P&L-based quality scores")

        economic_results = {}

        try:
            # Get target columns
            target_columns = [col for col in labels.columns if 'target' in col.lower()]

            if not target_columns:
                tprint_warning("⚠️ No target columns found for economic validation")
                return economic_results

            # Process each target
            for target_col in target_columns:
                tprint_info(f"📈 Assessing economic quality for target: {target_col}")

                # Extract target data
                target_labels = labels[target_col].dropna()
                target_confidence = confidence_scores.get(target_col, pd.Series(1.0, index=target_labels.index)) if confidence_scores is not None else pd.Series(1.0, index=target_labels.index)
                target_eligibility = eligibility_masks.get(target_col, pd.Series(True, index=target_labels.index)) if eligibility_masks is not None else pd.Series(True, index=target_labels.index)

                # Filter by eligibility
                eligible_mask = target_eligibility & target_eligibility.notna()
                if not eligible_mask.any():
                    tprint_warning(f"⚠️ No eligible samples for target {target_col}")
                    continue

                target_labels_eligible = target_labels[eligible_mask]
                target_confidence_eligible = target_confidence[eligible_mask]

                # Align with price data
                common_index = target_labels_eligible.index.intersection(prices.index)
                if len(common_index) < self.config.min_trades * 2:
                    tprint_warning(f"⚠️ Insufficient aligned data for target {target_col}: {len(common_index)} periods")
                    continue

                target_labels_aligned = target_labels_eligible.loc[common_index]
                target_confidence_aligned = target_confidence_eligible.loc[common_index]
                prices_aligned = prices.loc[common_index]

                # Simulate trading performance
                economic_metrics = self._simulate_trading_performance(
                    target_labels_aligned, target_confidence_aligned, prices_aligned, target_col
                )

                economic_results[target_col] = economic_metrics

        except Exception as e:
            tprint_error(f"❌ Economic validation failed: {e}")
            return economic_results

        processing_time = (datetime.now() - start_time).total_seconds()
        tprint_success("✅ Economic validation completed")
        tprint_info(f"   → Processing time: {processing_time:.2f}s")
        tprint_info(f"   → Targets assessed: {len(economic_results)}")

        return economic_results

    def _simulate_trading_performance(self, labels: pd.Series, confidence: pd.Series,
                                    prices: pd.DataFrame, target_name: str) -> EconomicMetrics:
        """Simulate trading performance for a single target."""
        try:
            # Initialize metrics
            metrics = EconomicMetrics(n_periods=len(labels))

            # Generate trading signals
            signals = self._generate_trading_signals(labels, confidence)

            # Simulate portfolio returns
            portfolio_returns = self._simulate_portfolio_returns(signals, prices, labels.index)

            if len(portfolio_returns) < self.config.warmup_periods + self.config.min_trades:
                tprint_warning(f"⚠️ Insufficient data for reliable metrics: {len(portfolio_returns)} periods")
                return metrics

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio_returns, signals, target_name)

            return metrics

        except Exception as e:
            tprint_warning(f"⚠️ Error simulating trading performance for {target_name}: {e}")
            return EconomicMetrics(n_periods=len(labels))

    def _generate_trading_signals(self, labels: pd.Series, confidence: pd.Series) -> pd.Series:
        """Generate trading signals from labels and confidence scores."""
        # Convert labels to position signals (-1, 0, 1)
        signals = pd.Series(0.0, index=labels.index)

        # Long signals for positive labels
        long_mask = labels > 0
        signals[long_mask] = confidence[long_mask]  # Scale by confidence

        # Short signals for negative labels (optional - depends on strategy)
        short_mask = labels < 0
        signals[short_mask] = -confidence[short_mask]  # Scale by confidence

        return signals

    def _simulate_portfolio_returns(self, signals: pd.Series, prices: pd.DataFrame,
                                  index: pd.Index) -> pd.Series:
        """Simulate portfolio returns including transaction costs."""
        try:
            # Calculate price returns
            price_returns = prices['close'].pct_change().fillna(0)

            # Initialize portfolio
            position = 0.0
            portfolio_value = self.config.initial_capital
            portfolio_returns = []

            # Transaction cost calculation
            total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps

            for i in range(len(signals)):
                current_signal = signals.iloc[i]
                current_price = prices['close'].iloc[i]

                # Calculate position size based on signal strength
                target_position = self._calculate_position_size(current_signal, portfolio_value, current_price)

                # Calculate transaction costs if position changes
                position_change = target_position - position
                if abs(position_change) > 1e-6:  # Significant position change
                    # Transaction cost as percentage of capital
                    transaction_cost = abs(position_change) * current_price * (total_cost_bps / 10000)

                    # Market impact (scales with position size)
                    market_impact = abs(position_change) * current_price * (self.config.market_impact_factor / 10000)
                    total_cost = transaction_cost + market_impact

                    portfolio_value -= total_cost

                # Update position
                position = target_position

                # Calculate period return (position * price_return - costs)
                if abs(position) > 1e-6:  # In position
                    period_return = position * price_returns.iloc[i]
                else:
                    period_return = 0.0

                # Store portfolio value and return
                portfolio_returns.append(period_return)
                portfolio_value *= (1 + period_return)

            return pd.Series(portfolio_returns, index=index)

        except Exception as e:
            tprint_warning(f"⚠️ Error in portfolio simulation: {e}")
            return pd.Series(dtype=float)

    def _calculate_position_size(self, signal: float, portfolio_value: float, price: float) -> float:
        """Calculate position size based on signal strength."""
        if abs(signal) < 1e-6:
            return 0.0

        # Base position size (fraction of portfolio)
        base_size = abs(signal) * self.config.max_position_size

        if self.config.position_size_method == "equal":
            # Equal position sizing
            return base_size

        elif self.config.position_size_method == "kelly":
            # Kelly criterion (simplified)
            # In practice, this would need historical win/loss data
            # For now, use a conservative Kelly approximation
            kelly_fraction = min(0.25, abs(signal) * 0.5)  # Conservative Kelly
            return kelly_fraction

        elif self.config.position_size_method == "volatility_scaled":
            # Volatility-adjusted position sizing
            # This would need volatility estimates - simplified for now
            vol_adjustment = 1.0  # Placeholder - would use actual volatility
            return base_size * vol_adjustment

        else:
            return base_size

    def _calculate_performance_metrics(self, portfolio_returns: pd.Series,
                                    signals: pd.Series, target_name: str) -> EconomicMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            metrics = EconomicMetrics(n_periods=len(portfolio_returns))

            if len(portfolio_returns) < self.config.warmup_periods:
                return metrics

            # Skip warmup period for metrics calculation
            returns = portfolio_returns.iloc[self.config.warmup_periods:]

            if len(returns) == 0:
                return metrics

            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-8)

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())

            # Trade-level metrics
            trade_returns = self._calculate_trade_returns(signals, returns)
            if len(trade_returns) >= self.config.min_trades:
                win_rate = (trade_returns > 0).sum() / len(trade_returns)
                profit_factor = abs(trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum()) if (trade_returns < 0).sum() > 0 else float('inf')

                # Individual trade statistics
                winning_trades = trade_returns[trade_returns > 0]
                losing_trades = trade_returns[trade_returns < 0]

                metrics.winning_trades = len(winning_trades)
                metrics.losing_trades = len(losing_trades)
                metrics.avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
                metrics.avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0.0
                metrics.largest_win = winning_trades.max() if len(winning_trades) > 0 else 0.0
                metrics.largest_loss = abs(losing_trades.min()) if len(losing_trades) > 0 else 0.0
                metrics.win_rate = win_rate
                metrics.profit_factor = profit_factor
            else:
                metrics.win_rate = 0.0
                metrics.profit_factor = 0.0

            # Advanced metrics
            calmar_ratio = sharpe_ratio / (max_drawdown + 1e-8) if max_drawdown > 0 else 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
            sortino_ratio = (returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0.0

            # Update metrics
            metrics.total_return = total_return
            metrics.volatility = volatility
            metrics.sharpe_ratio = sharpe_ratio
            metrics.max_drawdown = max_drawdown
            metrics.calmar_ratio = calmar_ratio
            metrics.sortino_ratio = sortino_ratio
            metrics.total_trades = len(trade_returns)

            # Economic quality score (weighted combination of key metrics)
            metrics.economic_quality_score = self._calculate_economic_quality_score(metrics)

            return metrics

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating performance metrics for {target_name}: {e}")
            return EconomicMetrics(n_periods=len(portfolio_returns))

    def _calculate_trade_returns(self, signals: pd.Series, returns: pd.Series) -> pd.Series:
        """Calculate individual trade returns."""
        try:
            trade_returns = []

            # Find trade entry and exit points
            position_changes = signals.diff().fillna(0)
            entry_points = position_changes != 0

            if not entry_points.any():
                return pd.Series(dtype=float)

            # For simplicity, calculate returns between position changes
            # In a more sophisticated implementation, this would track individual trades
            current_position = 0.0
            trade_start_idx = None

            for i in range(len(signals)):
                current_signal = signals.iloc[i]

                if abs(current_signal) > 1e-6 and current_position == 0.0:
                    # Trade entry
                    trade_start_idx = i
                    current_position = current_signal
                elif abs(current_signal) < 1e-6 and abs(current_position) > 1e-6:
                    # Trade exit
                    if trade_start_idx is not None:
                        trade_return = returns.iloc[trade_start_idx:i+1].sum() * current_position
                        trade_returns.append(trade_return)
                    trade_start_idx = None
                    current_position = 0.0

            # Close any open positions at the end
            if abs(current_position) > 1e-6 and trade_start_idx is not None:
                trade_return = returns.iloc[trade_start_idx:].sum() * current_position
                trade_returns.append(trade_return)

            return pd.Series(trade_returns)

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating trade returns: {e}")
            return pd.Series(dtype=float)

    def _calculate_economic_quality_score(self, metrics: EconomicMetrics) -> float:
        """Calculate composite economic quality score."""
        try:
            # Weighted combination of key economic metrics
            score = (
                0.3 * max(0, metrics.sharpe_ratio) / 2.0 +  # Normalize Sharpe (typical range 0-2)
                0.2 * (1.0 - min(1.0, metrics.max_drawdown)) +  # Penalize drawdown
                0.2 * min(1.0, metrics.win_rate) +  # Win rate (0-1)
                0.15 * min(2.0, metrics.profit_factor) / 2.0 +  # Profit factor (normalize 0-2)
                0.15 * min(1.0, max(0, metrics.total_return))  # Total return cap at 100%
            )

            return min(1.0, score)

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating economic quality score: {e}")
            return 0.0


def create_economic_validator(config: Optional[EconomicConfig] = None) -> EconomicValidator:
    """Create economic validator with specified configuration."""
    return EconomicValidator(config)


def validate_labels_economically(labels: pd.DataFrame, prices: pd.DataFrame,
                               config: Optional[EconomicConfig] = None) -> Dict[str, EconomicMetrics]:
    """Validate labels economically with default configuration."""
    validator = EconomicValidator(config)
    return validator.calculate_pnl_quality(labels, prices)