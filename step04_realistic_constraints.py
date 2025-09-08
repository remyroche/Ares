#!/usr/bin/env python3
"""
Step04 Realistic Trading Constraints and Transaction Costs

This module addresses the issue of theoretical returns being meaningless without
realistic constraints by implementing comprehensive trading cost modeling and
realistic trading constraints.

Features:
- Realistic transaction cost modeling
- Market impact and slippage estimation
- Position sizing constraints
- Risk management rules
- Portfolio-level constraints
- Liquidity constraints
- Regulatory constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import collections
import time

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradingConstraints:
    """Trading constraints configuration."""
    
    # Position sizing constraints
    max_position_size: float = 0.1  # 10% of portfolio
    min_position_size: float = 0.001  # 0.1% of portfolio
    max_leverage: float = 1.0  # No leverage
    
    # Risk management constraints
    max_drawdown: float = 0.2  # 20% maximum drawdown
    max_daily_loss: float = 0.05  # 5% maximum daily loss
    max_correlation: float = 0.7  # Maximum correlation between positions
    
    # Transaction cost constraints
    base_commission_bps: float = 2.0  # 2 basis points base commission
    market_impact_bps: float = 1.0  # 1 basis point market impact
    slippage_bps: float = 0.5  # 0.5 basis points slippage
    spread_bps: float = 2.0  # 2 basis points bid-ask spread
    
    # Liquidity constraints
    min_volume_threshold: float = 1000000  # $1M minimum daily volume
    max_trade_size_ratio: float = 0.01  # Max 1% of daily volume per trade
    
    # Regulatory constraints
    max_positions: int = 50  # Maximum number of positions
    min_holding_period: int = 1  # Minimum holding period in periods
    max_turnover: float = 10.0  # Maximum annual turnover (10x)
    
    # Market hours constraints
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    timezone: str = "US/Eastern"

@dataclass
class TransactionCosts:
    """Transaction cost breakdown."""
    
    commission: float = 0.0
    market_impact: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0
    total_cost: float = 0.0
    cost_bps: float = 0.0

class RealisticTradingSimulator:
    """
    Realistic trading simulator with comprehensive constraints and cost modeling.
    
    This simulator addresses the issue of theoretical returns by implementing
    realistic trading conditions that would be encountered in live trading.
    """
    
    def __init__(self, constraints: TradingConstraints, initial_capital: float = 1000000):
        self.constraints = constraints
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.positions = {}  # {symbol: {'quantity': float, 'avg_price': float, 'entry_time': datetime}}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.total_volume = 0.0
        
        # Performance tracking
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.daily_returns = []
        self.trade_history = []
        
        self.logger.info("✅ Realistic Trading Simulator initialized")
        self.logger.info(f"   Initial capital: ${initial_capital:,.2f}")
        self.logger.info(f"   Max position size: {constraints.max_position_size:.1%}")
        self.logger.info(f"   Max drawdown: {constraints.max_drawdown:.1%}")
    
    def calculate_transaction_costs(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        side: OrderSide,
        market_data: Dict[str, Any]
    ) -> TransactionCosts:
        """
        Calculate realistic transaction costs for a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            side: Buy or sell
            market_data: Current market data (volume, spread, etc.)
            
        Returns:
            Detailed transaction cost breakdown
        """
        
        # Base commission
        commission = abs(quantity * price * self.constraints.base_commission_bps / 10000)
        
        # Market impact (depends on trade size relative to volume)
        daily_volume = market_data.get('volume', 1000000)
        trade_size = abs(quantity * price)
        volume_ratio = trade_size / daily_volume
        
        # Market impact increases with trade size
        market_impact_multiplier = min(volume_ratio / self.constraints.max_trade_size_ratio, 3.0)
        market_impact = trade_size * self.constraints.market_impact_bps / 10000 * market_impact_multiplier
        
        # Slippage (random component)
        slippage = trade_size * self.constraints.slippage_bps / 10000 * np.random.uniform(0.5, 1.5)
        
        # Spread cost (half the spread for each trade)
        spread_cost = trade_size * self.constraints.spread_bps / 10000 / 2
        
        # Total costs
        total_cost = commission + market_impact + slippage + spread_cost
        cost_bps = (total_cost / trade_size) * 10000 if trade_size > 0 else 0
        
        return TransactionCosts(
            commission=commission,
            market_impact=market_impact,
            slippage=slippage,
            spread_cost=spread_cost,
            total_cost=total_cost,
            cost_bps=cost_bps
        )
    
    def validate_trade(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        side: OrderSide,
        market_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a trade meets all constraints.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Position sizing constraints
        trade_value = abs(quantity * price)
        position_ratio = trade_value / self.portfolio_value
        
        if position_ratio > self.constraints.max_position_size:
            violations.append(f"Position size {position_ratio:.1%} exceeds maximum {self.constraints.max_position_size:.1%}")
        
        if position_ratio < self.constraints.min_position_size and trade_value > 0:
            violations.append(f"Position size {position_ratio:.1%} below minimum {self.constraints.min_position_size:.1%}")
        
        # Liquidity constraints
        daily_volume = market_data.get('volume', 0)
        if daily_volume < self.constraints.min_volume_threshold:
            violations.append(f"Daily volume ${daily_volume:,.0f} below minimum ${self.constraints.min_volume_threshold:,.0f}")
        
        volume_ratio = trade_value / daily_volume if daily_volume > 0 else 0
        if volume_ratio > self.constraints.max_trade_size_ratio:
            violations.append(f"Trade size {volume_ratio:.1%} of daily volume exceeds maximum {self.constraints.max_trade_size_ratio:.1%}")
        
        # Maximum positions constraint
        if symbol not in self.positions and len(self.positions) >= self.constraints.max_positions:
            violations.append(f"Maximum positions {self.constraints.max_positions} reached")
        
        # Risk management constraints
        if self.max_drawdown >= self.constraints.max_drawdown:
            violations.append(f"Maximum drawdown {self.max_drawdown:.1%} reached")
        
        if self.daily_pnl <= -self.constraints.max_daily_loss * self.portfolio_value:
            violations.append(f"Daily loss limit {self.constraints.max_daily_loss:.1%} reached")
        
        # Minimum holding period
        if symbol in self.positions:
            current_time = market_data.get('timestamp', datetime.now())
            entry_time = self.positions[symbol]['entry_time']
            holding_period = (current_time - entry_time).total_seconds() / 60  # minutes
            
            if holding_period < self.constraints.min_holding_period:
                violations.append(f"Minimum holding period {self.constraints.min_holding_period} minutes not met")
        
        return len(violations) == 0, violations
    
    def execute_trade(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        side: OrderSide,
        market_data: Dict[str, Any],
        timestamp: datetime = None
    ) -> Dict[str, Any]:
        """
        Execute a trade with realistic constraints and costs.
        
        Returns:
            Trade execution result
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate trade
        is_valid, violations = self.validate_trade(symbol, quantity, price, side, market_data)
        
        if not is_valid:
            return {
                'success': False,
                'violations': violations,
                'message': f"Trade rejected: {', '.join(violations)}"
            }
        
        # Calculate transaction costs
        costs = self.calculate_transaction_costs(symbol, quantity, price, side, market_data)
        
        # Calculate effective price (including costs)
        if side == OrderSide.BUY:
            effective_price = price + (costs.total_cost / abs(quantity))
        else:
            effective_price = price - (costs.total_cost / abs(quantity))
        
        # Update portfolio
        trade_value = quantity * effective_price
        
        if side == OrderSide.BUY:
            # Buying
            if symbol in self.positions:
                # Add to existing position
                current_quantity = self.positions[symbol]['quantity']
                current_avg_price = self.positions[symbol]['avg_price']
                
                new_quantity = current_quantity + quantity
                new_avg_price = ((current_quantity * current_avg_price) + (quantity * effective_price)) / new_quantity
                
                self.positions[symbol] = {
                    'quantity': new_quantity,
                    'avg_price': new_avg_price,
                    'entry_time': self.positions[symbol]['entry_time']
                }
            else:
                # New position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': effective_price,
                    'entry_time': timestamp
                }
            
            self.cash -= trade_value + costs.total_cost
            
        else:
            # Selling
            if symbol not in self.positions:
                return {
                    'success': False,
                    'message': f"Cannot sell {symbol}: position not found"
                }
            
            current_quantity = self.positions[symbol]['quantity']
            
            if abs(quantity) > current_quantity:
                return {
                    'success': False,
                    'message': f"Cannot sell {abs(quantity)} {symbol}: only {current_quantity} available"
                }
            
            # Update position
            new_quantity = current_quantity + quantity  # quantity is negative for sell
            
            if abs(new_quantity) < 1e-8:  # Position closed
                del self.positions[symbol]
            else:
                self.positions[symbol]['quantity'] = new_quantity
            
            self.cash += trade_value - costs.total_cost
        
        # Update portfolio metrics
        self.total_trades += 1
        self.total_volume += abs(trade_value)
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'price': price,
            'effective_price': effective_price,
            'trade_value': trade_value,
            'costs': costs,
            'cash_after': self.cash,
            'positions_after': len(self.positions)
        }
        
        self.trade_history.append(trade_record)
        
        # Update portfolio value
        self._update_portfolio_value(market_data)
        
        return {
            'success': True,
            'trade_record': trade_record,
            'message': f"Trade executed: {side.value} {abs(quantity)} {symbol} at {effective_price:.4f}"
        }
    
    def _update_portfolio_value(self, market_data: Dict[str, Any]):
        """Update portfolio value and performance metrics."""
        
        # Calculate current portfolio value
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            # Use current market price (would be from market_data in real implementation)
            current_price = market_data.get(f'{symbol}_price', position['avg_price'])
            positions_value += position['quantity'] * current_price
        
        self.portfolio_value = self.cash + positions_value
        
        # Update peak value and drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate daily return
        if len(self.daily_returns) > 0:
            daily_return = (self.portfolio_value / self.initial_capital) - 1
            self.daily_returns.append(daily_return)
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics with realistic constraints."""
        
        if len(self.daily_returns) == 0:
            return {'error': 'No trading data available'}
        
        returns = np.array(self.daily_returns)
        
        # Basic metrics
        total_return = (self.portfolio_value / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Trading metrics
        turnover = self.total_volume / self.initial_capital
        avg_trade_size = self.total_volume / self.total_trades if self.total_trades > 0 else 0
        
        # Cost analysis
        total_costs = sum(trade['costs'].total_cost for trade in self.trade_history)
        cost_ratio = total_costs / self.initial_capital
        
        # Win rate
        profitable_trades = 0
        for i in range(1, len(self.trade_history)):
            if self.trade_history[i]['trade_value'] > 0:  # Simplified profit check
                profitable_trades += 1
        
        win_rate = profitable_trades / len(self.trade_history) if len(self.trade_history) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': (self.peak_value - self.portfolio_value) / self.peak_value,
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'turnover': turnover,
            'avg_trade_size': avg_trade_size,
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'win_rate': win_rate,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_count': len(self.positions),
            'constraints_violated': self._check_constraint_violations()
        }
    
    def _check_constraint_violations(self) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        # Check drawdown constraint
        if self.max_drawdown > self.constraints.max_drawdown:
            violations.append(f"Max drawdown {self.max_drawdown:.1%} > {self.constraints.max_drawdown:.1%}")
        
        # Check turnover constraint
        turnover = self.total_volume / self.initial_capital
        if turnover > self.constraints.max_turnover:
            violations.append(f"Turnover {turnover:.1f}x > {self.constraints.max_turnover:.1f}x")
        
        # Check position count constraint
        if len(self.positions) > self.constraints.max_positions:
            violations.append(f"Positions {len(self.positions)} > {self.constraints.max_positions}")
        
        return violations
    
    def simulate_trading_signals(
        self, 
        signals: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Simulate trading based on signals with realistic constraints.
        
        Args:
            signals: DataFrame with trading signals
            market_data: DataFrame with market data (OHLCV)
            
        Returns:
            Simulation results
        """
        self.logger.info(f"🎯 Starting trading simulation with {len(signals)} signals")
        
        simulation_results = {
            'trades_executed': 0,
            'trades_rejected': 0,
            'rejection_reasons': {},
            'performance_metrics': {},
            'trade_history': []
        }
        
        for idx, signal_row in signals.iterrows():
            symbol = signal_row.get('symbol', 'DEFAULT')
            signal = signal_row.get('label', 0)
            timestamp = signal_row.get('timestamp', datetime.now())
            
            if signal == 0:  # No signal
                continue
            
            # Get current market data
            current_market = {
                'volume': market_data.iloc[idx].get('volume', 1000000),
                'price': market_data.iloc[idx].get('close', 100),
                'timestamp': timestamp
            }
            
            # Determine trade parameters
            if signal == 1:  # Buy signal
                side = OrderSide.BUY
                quantity = self._calculate_position_size(symbol, current_market['price'])
            elif signal == -1:  # Sell signal
                side = OrderSide.SELL
                quantity = -self._calculate_position_size(symbol, current_market['price'])
            else:
                continue
            
            # Execute trade
            result = self.execute_trade(
                symbol=symbol,
                quantity=quantity,
                price=current_market['price'],
                side=side,
                market_data=current_market,
                timestamp=timestamp
            )
            
            if result['success']:
                simulation_results['trades_executed'] += 1
                simulation_results['trade_history'].append(result['trade_record'])
            else:
                simulation_results['trades_rejected'] += 1
                reason = result.get('message', 'Unknown reason')
                simulation_results['rejection_reasons'][reason] = simulation_results['rejection_reasons'].get(reason, 0) + 1
        
        # Calculate final performance metrics
        simulation_results['performance_metrics'] = self.calculate_performance_metrics()
        
        self.logger.info(f"✅ Simulation completed")
        self.logger.info(f"   Trades executed: {simulation_results['trades_executed']}")
        self.logger.info(f"   Trades rejected: {simulation_results['trades_rejected']}")
        self.logger.info(f"   Final portfolio value: ${self.portfolio_value:,.2f}")
        
        return simulation_results
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on constraints."""
        
        # Use a fraction of max position size for risk management
        target_position_value = self.portfolio_value * self.constraints.max_position_size * 0.5
        
        # Calculate quantity
        quantity = target_position_value / price
        
        # Round to reasonable precision
        return round(quantity, 2)


# Example usage and testing
def test_realistic_trading_simulator():
    """Test the realistic trading simulator."""
    
    # Create sample market data
    n_samples = 1000
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, n_samples)
    })
    
    # Create sample trading signals
    signals = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'TEST',
        'label': np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.8, 0.1])
    })
    
    # Create trading constraints
    constraints = TradingConstraints(
        max_position_size=0.05,  # 5% max position
        max_drawdown=0.15,       # 15% max drawdown
        base_commission_bps=2.0, # 2 bps commission
        market_impact_bps=1.0,   # 1 bp market impact
        slippage_bps=0.5,        # 0.5 bp slippage
        spread_bps=2.0,          # 2 bp spread
        min_volume_threshold=500000,  # $500K min volume
        max_trade_size_ratio=0.005,   # 0.5% of daily volume
        max_positions=10,        # Max 10 positions
        min_holding_period=5,    # 5 minute minimum holding
        max_turnover=5.0         # 5x annual turnover
    )
    
    # Initialize simulator
    simulator = RealisticTradingSimulator(constraints, initial_capital=1000000)
    
    # Run simulation
    print("=== Testing Realistic Trading Simulator ===")
    results = simulator.simulate_trading_signals(signals, market_data)
    
    print(f"Trades executed: {results['trades_executed']}")
    print(f"Trades rejected: {results['trades_rejected']}")
    print(f"Rejection reasons: {results['rejection_reasons']}")
    
    # Performance metrics
    metrics = results['performance_metrics']
    print(f"\nPerformance Metrics:")
    print(f"Total return: {metrics['total_return']:.2%}")
    print(f"Annualized return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total costs: ${metrics['total_costs']:,.2f}")
    print(f"Cost ratio: {metrics['cost_ratio']:.2%}")
    print(f"Turnover: {metrics['turnover']:.1f}x")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    
    if metrics['constraints_violated']:
        print(f"Constraint violations: {metrics['constraints_violated']}")
    
    return results


if __name__ == "__main__":
    test_realistic_trading_simulator()