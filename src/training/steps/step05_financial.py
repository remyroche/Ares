from ..standardized_parquet_handler import standardized_parquet_handler
"""
Step05 Financial Calculations Module

This module provides comprehensive financial calculations for Step05 labeling,
including transaction cost modeling, risk-adjusted returns, and trading strategy
performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates
import logging
import time

logger = system_logger.getChild('Step05Financial')

@dataclass
class TransactionCosts:
    """Transaction cost parameters for financial calculations."""
    maker_fee: float = 0.001  # 0.1% maker fee
    taker_fee: float = 0.001  # 0.1% taker fee
    slippage_bps: float = 2.0  # 2 basis points slippage
    funding_rate: float = 0.0001  # 0.01% funding rate (for perpetuals)
    min_trade_size: float = 10.0  # Minimum trade size in USD
    max_trade_size: float = 100000.0  # Maximum trade size in USD

@dataclass
class TradingPerformance:
    """Trading performance metrics."""
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

@dataclass
class RiskMetrics:
    """Risk assessment metrics."""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    volatility: float
    beta: float
    correlation_to_market: float
    tail_risk: float
    downside_deviation: float

class Step05FinancialCalculator:
    """Comprehensive financial calculator for Step05 labeling results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.transaction_costs = TransactionCosts()
        self._load_transaction_cost_config()
        
    def _load_transaction_cost_config(self):
        """Load transaction cost configuration from config."""
        if 'transaction_costs' in self.config:
            tc_config = self.config['transaction_costs']
            self.transaction_costs = TransactionCosts(
                maker_fee=tc_config.get('maker_fee', 0.001),
                taker_fee=tc_config.get('taker_fee', 0.001),
                slippage_bps=tc_config.get('slippage_bps', 2.0),
                funding_rate=tc_config.get('funding_rate', 0.0001),
                min_trade_size=tc_config.get('min_trade_size', 10.0),
                max_trade_size=tc_config.get('max_trade_size', 100000.0)
            )
    
    @traced(span_name='calculate_transaction_costs')
    @validates()
    @handles_errors()
    def calculate_transaction_costs(self, data: pd.DataFrame, 
                                 position_sizes: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate transaction costs for each trade based on labels.
        
        Args:
            data: DataFrame with price data and labels
            position_sizes: Optional position sizes for each trade
            
        Returns:
            Series with transaction costs for each period
        """
        try:
            self.logger.info("💰 Calculating transaction costs...")
            
            if 'label' not in data.columns or 'close' not in data.columns:
                self.logger.warning("⚠️ Missing required columns for transaction cost calculation")
                return pd.Series(0.0, index=data.index)
            
            # Default position size if not provided
            if position_sizes is None:
                position_sizes = pd.Series(1000.0, index=data.index)  # $1000 default
            
            transaction_costs = pd.Series(0.0, index=data.index)
            
            # Calculate costs for each trade
            for i in range(len(data)):
                if pd.isna(data['label'].iloc[i]) or data['label'].iloc[i] == 0:
                    continue
                
                position_size = position_sizes.iloc[i]
                price = data['close'].iloc[i]
                
                # Skip if position size is too small
                if position_size < self.transaction_costs.min_trade_size:
                    continue
                
                # Cap position size
                position_size = min(position_size, self.transaction_costs.max_trade_size)
                
                # Calculate different cost components
                costs = self._calculate_trade_costs(position_size, price, data['label'].iloc[i])
                transaction_costs.iloc[i] = costs
            
            total_costs = transaction_costs.sum()
            self.logger.info(f"✅ Transaction costs calculated. Total: ${total_costs:.2f}")
            
            return transaction_costs
            
        except Exception as e:
            self.logger.error(f"❌ Transaction cost calculation failed: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _calculate_trade_costs(self, position_size: float, price: float, label: int) -> float:
        """Calculate costs for a single trade."""
        try:
            # Base trading fee (assume taker fee for simplicity)
            trading_fee = position_size * self.transaction_costs.taker_fee
            
            # Slippage cost
            slippage_cost = position_size * (self.transaction_costs.slippage_bps / 10000)
            
            # Funding cost (for perpetuals, assume 8-hour funding)
            funding_cost = position_size * self.transaction_costs.funding_rate * (8/24)
            
            # Market impact (simplified model based on position size)
            market_impact = self._calculate_market_impact(position_size, price)
            
            total_cost = trading_fee + slippage_cost + funding_cost + market_impact
            
            return total_cost
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating trade costs: {e}")
            return position_size * 0.002  # 0.2% fallback cost
    
    def _calculate_market_impact(self, position_size: float, price: float) -> float:
        """Calculate market impact cost based on position size."""
        try:
            # Simplified market impact model
            # Larger positions have higher market impact
            size_ratio = position_size / self.transaction_costs.max_trade_size
            
            # Market impact increases quadratically with size
            impact_rate = 0.0001 * (size_ratio ** 2)  # 0.01% max impact
            
            return position_size * impact_rate
            
        except Exception:
            return 0.0
    
    @traced(span_name='calculate_trading_performance')
    @validates()
    @handles_errors()
    def calculate_trading_performance(self, data: pd.DataFrame,
                                   transaction_costs: Optional[pd.Series] = None) -> TradingPerformance:
        """
        Calculate comprehensive trading performance metrics.
        
        Args:
            data: DataFrame with price data and labels
            transaction_costs: Optional transaction costs series
            
        Returns:
            TradingPerformance object with all metrics
        """
        try:
            self.logger.info("📊 Calculating trading performance metrics...")
            
            if 'label' not in data.columns or 'close' not in data.columns:
                raise ValueError("Missing required columns: 'label' and 'close'")
            
            # Calculate transaction costs if not provided
            if transaction_costs is None:
                transaction_costs = self.calculate_transaction_costs(data)
            
            # Generate trade returns
            trade_returns = self._generate_trade_returns(data, transaction_costs)
            
            # Calculate basic metrics
            total_return = trade_returns.sum()
            total_trades = len(trade_returns[trade_returns != 0])
            winning_trades = len(trade_returns[trade_returns > 0])
            losing_trades = len(trade_returns[trade_returns < 0])
            
            # Calculate win rate and profit factor
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            gross_profit = trade_returns[trade_returns > 0].sum()
            gross_loss = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate average win/loss
            avg_win = trade_returns[trade_returns > 0].mean() if winning_trades > 0 else 0.0
            avg_loss = trade_returns[trade_returns < 0].mean() if losing_trades > 0 else 0.0
            
            # Calculate largest win/loss
            largest_win = trade_returns.max() if len(trade_returns) > 0 else 0.0
            largest_loss = trade_returns.min() if len(trade_returns) > 0 else 0.0
            
            # Calculate risk metrics
            volatility = trade_returns.std() if len(trade_returns) > 1 else 0.0
            sharpe_ratio = trade_returns.mean() / volatility if volatility > 0 else 0.0
            
            # Calculate Sortino ratio (downside deviation)
            downside_returns = trade_returns[trade_returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0.0
            sortino_ratio = trade_returns.mean() / downside_deviation if downside_deviation > 0 else 0.0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + trade_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate annualized return (assuming daily data)
            periods_per_year = 252  # Trading days
            annualized_return = (1 + total_return) ** (periods_per_year / len(trade_returns)) - 1 if len(trade_returns) > 0 else 0.0
            
            # Calculate average holding period
            avg_holding_period = self._calculate_avg_holding_period(data)
            
            # Calculate transaction cost impact
            total_transaction_costs = transaction_costs.sum()
            cost_impact = total_transaction_costs / abs(total_return) if total_return != 0 else 0.0
            net_return = total_return - total_transaction_costs
            
            performance = TradingPerformance(
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
                cost_impact=cost_impact
            )
            
            self.logger.info(f"✅ Trading performance calculated. Net return: {net_return:.2%}")
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Trading performance calculation failed: {e}")
            # Return default performance metrics
            return TradingPerformance(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, profit_factor=0.0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0, avg_holding_period=0.0,
                transaction_costs=0.0, net_return=0.0, cost_impact=0.0
            )
    
    def _generate_trade_returns(self, data: pd.DataFrame, 
                              transaction_costs: pd.Series) -> pd.Series:
        """Generate trade returns based on labels and price movements."""
        try:
            returns = pd.Series(0.0, index=data.index)
            
            # Simple trade simulation
            for i in range(len(data) - 1):
                if pd.isna(data['label'].iloc[i]) or data['label'].iloc[i] == 0:
                    continue
                
                # Calculate return for this trade
                entry_price = data['close'].iloc[i]
                exit_price = data['close'].iloc[i + 1]  # Simplified: exit next period
                
                # Calculate gross return
                if data['label'].iloc[i] == 1:  # Long position
                    gross_return = (exit_price - entry_price) / entry_price
                elif data['label'].iloc[i] == -1:  # Short position
                    gross_return = (entry_price - exit_price) / entry_price
                else:
                    gross_return = 0.0
                
                # Subtract transaction costs
                net_return = gross_return - (transaction_costs.iloc[i] / 1000)  # Assume $1000 position
                returns.iloc[i] = net_return
            
            return returns
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating trade returns: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _calculate_avg_holding_period(self, data: pd.DataFrame) -> float:
        """Calculate average holding period for trades."""
        try:
            if 'label' not in data.columns:
                return 0.0
            
            holding_periods = []
            current_position = None
            entry_time = None
            
            for i, label in enumerate(data['label']):
                if pd.isna(label):
                    continue
                
                if label != 0 and current_position is None:
                    # Enter position
                    current_position = label
                    entry_time = i
                elif label == 0 and current_position is not None:
                    # Exit position
                    holding_period = i - entry_time
                    holding_periods.append(holding_period)
                    current_position = None
                    entry_time = None
            
            return np.mean(holding_periods) if holding_periods else 0.0
            
        except Exception:
            return 0.0
    
    @traced(span_name='calculate_risk_metrics')
    @validates()
    @handles_errors()
    def calculate_risk_metrics(self, data: pd.DataFrame,
                            market_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            data: DataFrame with price data and labels
            market_data: Optional market benchmark data
            
        Returns:
            RiskMetrics object with all risk measures
        """
        try:
            self.logger.info("⚠️ Calculating risk metrics...")
            
            if 'close' not in data.columns:
                raise ValueError("Missing required column: 'close'")
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                return self._default_risk_metrics()
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5)  # 5th percentile (95% VaR)
            var_99 = np.percentile(returns, 1)  # 1st percentile (99% VaR)
            
            # Calculate Expected Shortfall (Conditional VaR)
            expected_shortfall = returns[returns <= var_95].mean()
            
            # Calculate volatility
            volatility = returns.std()
            
            # Calculate downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0.0
            
            # Calculate tail risk (kurtosis)
            tail_risk = returns.kurtosis()
            
            # Calculate beta and correlation if market data provided
            beta = 0.0
            correlation_to_market = 0.0
            
            if market_data is not None and 'close' in market_data.columns:
                market_returns = market_data['close'].pct_change().dropna()
                
                # Align returns
                aligned_returns, aligned_market = returns.align(market_returns, join='inner')
                
                if len(aligned_returns) > 1 and len(aligned_market) > 1:
                    # Calculate correlation
                    correlation_to_market = aligned_returns.corr(aligned_market)
                    
                    # Calculate beta
                    covariance = np.cov(aligned_returns, aligned_market)[0, 1]
                    market_variance = aligned_market.var()
                    beta = covariance / market_variance if market_variance > 0 else 0.0
            
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                volatility=volatility,
                beta=beta,
                correlation_to_market=correlation_to_market,
                tail_risk=tail_risk,
                downside_deviation=downside_deviation
            )
            
            self.logger.info(f"✅ Risk metrics calculated. VaR 95%: {var_95:.2%}")
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Risk metrics calculation failed: {e}")
            return self._default_risk_metrics()
    
    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when calculation fails."""
        return RiskMetrics(
            var_95=0.0, var_99=0.0, expected_shortfall=0.0, volatility=0.0,
            beta=0.0, correlation_to_market=0.0, tail_risk=0.0, downside_deviation=0.0
        )
    
    @traced(span_name='calculate_position_sizing')
    @validates()
    @handles_errors()
    def calculate_position_sizing(self, data: pd.DataFrame,
                                confidence_scores: Optional[pd.Series] = None,
                                risk_budget: float = 0.02) -> pd.Series:
        """
        Calculate optimal position sizes using Kelly criterion and risk budgeting.
        
        Args:
            data: DataFrame with price data and labels
            confidence_scores: Optional confidence scores for each label
            risk_budget: Maximum risk per trade (default 2%)
            
        Returns:
            Series with recommended position sizes
        """
        try:
            self.logger.info("📏 Calculating position sizing...")
            
            if 'label' not in data.columns:
                return pd.Series(0.0, index=data.index)
            
            # Default confidence scores if not provided
            if confidence_scores is None:
                confidence_scores = pd.Series(0.5, index=data.index)
            
            position_sizes = pd.Series(0.0, index=data.index)
            
            # Calculate Kelly criterion for each trade
            for i in range(len(data)):
                if pd.isna(data['label'].iloc[i]) or data['label'].iloc[i] == 0:
                    continue
                
                confidence = confidence_scores.iloc[i] if i < len(confidence_scores) else 0.5
                
                # Simplified Kelly calculation
                # Kelly = (bp - q) / b, where b = odds, p = win probability, q = loss probability
                win_prob = confidence
                loss_prob = 1 - win_prob
                
                # Estimate odds from historical data (simplified)
                odds = 1.5  # Assume 1.5:1 odds
                
                kelly_fraction = (odds * win_prob - loss_prob) / odds
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                
                # Apply risk budget
                position_size = kelly_fraction * risk_budget * 10000  # Scale to dollar amount
                position_sizes.iloc[i] = position_size
            
            self.logger.info(f"✅ Position sizing calculated. Avg size: ${position_sizes.mean():.2f}")
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing calculation failed: {e}")
            return pd.Series(1000.0, index=data.index)  # Default $1000 position
    
    @traced(span_name='generate_financial_report')
    @validates()
    @handles_errors()
    def generate_financial_report(self, data: pd.DataFrame,
                                performance: TradingPerformance,
                                risk_metrics: RiskMetrics,
                                transaction_costs: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive financial report.
        
        Args:
            data: Original data
            performance: Trading performance metrics
            risk_metrics: Risk assessment metrics
            transaction_costs: Transaction costs series
            
        Returns:
            Dictionary with comprehensive financial analysis
        """
        try:
            self.logger.info("📊 Generating financial report...")
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'trading_performance': {
                    'total_return': performance.total_return,
                    'annualized_return': performance.annualized_return,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'sortino_ratio': performance.sortino_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'win_rate': performance.win_rate,
                    'profit_factor': performance.profit_factor,
                    'total_trades': performance.total_trades,
                    'net_return': performance.net_return,
                    'cost_impact': performance.cost_impact
                },
                'risk_metrics': {
                    'var_95': risk_metrics.var_95,
                    'var_99': risk_metrics.var_99,
                    'expected_shortfall': risk_metrics.expected_shortfall,
                    'volatility': risk_metrics.volatility,
                    'beta': risk_metrics.beta,
                    'correlation_to_market': risk_metrics.correlation_to_market,
                    'tail_risk': risk_metrics.tail_risk
                },
                'transaction_costs': {
                    'total_costs': transaction_costs.sum(),
                    'avg_cost_per_trade': transaction_costs.mean(),
                    'cost_distribution': {
                        'min': transaction_costs.min(),
                        'max': transaction_costs.max(),
                        'median': transaction_costs.median(),
                        'std': transaction_costs.std()
                    }
                },
                'recommendations': self._generate_financial_recommendations(performance, risk_metrics),
                'warnings': self._generate_financial_warnings(performance, risk_metrics)
            }
            
            self.logger.info("✅ Financial report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Financial report generation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_financial_recommendations(self, performance: TradingPerformance,
                                          risk_metrics: RiskMetrics) -> List[str]:
        """Generate financial recommendations based on performance and risk."""
        recommendations = []
        
        # Performance-based recommendations
        if performance.sharpe_ratio < 1.0:
            recommendations.append("Consider improving risk-adjusted returns - Sharpe ratio below 1.0")
        
        if performance.max_drawdown < -0.2:
            recommendations.append("High maximum drawdown detected - consider risk management improvements")
        
        if performance.cost_impact > 0.1:
            recommendations.append("High transaction cost impact - consider reducing trade frequency or improving execution")
        
        # Risk-based recommendations
        if risk_metrics.volatility > 0.3:
            recommendations.append("High volatility detected - consider position sizing adjustments")
        
        if abs(risk_metrics.beta) > 1.5:
            recommendations.append("High beta detected - consider market exposure management")
        
        if risk_metrics.tail_risk > 3.0:
            recommendations.append("High tail risk detected - consider tail risk hedging")
        
        return recommendations
    
    def _generate_financial_warnings(self, performance: TradingPerformance,
                                   risk_metrics: RiskMetrics) -> List[str]:
        """Generate financial warnings based on performance and risk."""
        warnings = []
        
        # Performance warnings
        if performance.total_return < 0:
            warnings.append("Negative total return - strategy may not be profitable")
        
        if performance.win_rate < 0.4:
            warnings.append("Low win rate - consider improving signal quality")
        
        if performance.profit_factor < 1.2:
            warnings.append("Low profit factor - risk/reward ratio may be unfavorable")
        
        # Risk warnings
        if risk_metrics.var_95 < -0.05:
            warnings.append("High VaR 95% - significant downside risk")
        
        if risk_metrics.expected_shortfall < -0.1:
            warnings.append("High expected shortfall - severe tail risk")
        
        return warnings