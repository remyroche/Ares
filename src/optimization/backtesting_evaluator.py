# src/optimization/backtesting_evaluator.py

"""
Backtesting Evaluator for Parameter Optimization
Provides realistic performance evaluation during parameter optimization.
"""

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


import class BacktestingEvaluator:
class BacktestingEvaluator:
    """
    Backtesting evaluator for parameter optimization.
    Simulates trading performance with given parameters.
    """

    def __init__(self, config: Dict[str, Any]):
    pass
    pass
    pass
        self.config = config
        self.logger = system_logger.getChild("BacktestingEvaluator")

        # Backtesting configuration
        self.initial_capital = config.get("backtesting", {}).get("initial_capital", 10000)
        self.commission_rate = config.get("backtesting", {}).get("commission_rate", 0.001)
        self.slippage = config.get("backtesting", {}).get("slippage", 0.0005)

        # Performance metrics weights
        self.metric_weights = {
            "sharpe_ratio": 0.3,
            "profit_factor": 0.25,
            "max_drawdown": 0.2,
            "win_rate": 0.15,
            "total_return": 0.1
        }

        # Mock market data (in practice, load real data)
        self.market_data = self._generate_mock_market_data()

    def _generate_mock_market_data(self) -> pd.DataFrame:
    pass
    pass
    pass
        """Generate mock market data for backtesting."""
        np.random.seed(42)

        # Generate 1000 data points
        n_points = 1000
        dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')

        # Generate price data with realistic patterns
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift, 2% volatility
        prices = [base_price]

        for ret in returns[1:]:
    pass
    pass
    pass
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
    pass
    pass
    pass
            # Generate realistic OHLC from close price
            volatility = np.random.uniform(0.005, 0.02)
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            open_price = np.random.uniform(low, high)
            volume = np.random.uniform(1000, 10000)

            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        return pd.DataFrame(data)

    @handle_errors(
        exceptions=(Exception,),
        default_return=0.0,
        context="backtesting evaluation"
    )
    async def evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """
        Evaluate parameters using backtesting simulation.

        Args:
            params: Parameter dictionary

        Returns:
            float: Performance score (higher is better)
        """
        try:
            # Run backtesting simulation
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            backtest_results = await self._run_backtest(params)

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(backtest_results)

            # Calculate weighted score
            score = self._calculate_weighted_score(metrics)

            return score

        except Exception as e:
            self.logger.error(f"Backtesting evaluation error: {e}")
            return 0.0

    async def _run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtesting simulation with given parameters."""
        try:
            # Initialize backtesting state
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            capital = self.initial_capital
            position = 0.0
            trades = []
            equity_curve = []

            # Strategy state
            in_position = False
            entry_price = 0.0
            entry_time = None

            # Process each data point
            for i, row in self.market_data.iterrows():
    pass
    pass
    pass
                current_price = row['close']
                current_time = row['timestamp']

                # Generate trading signals based on parameters
                signal = self._generate_signal(row, params, i)

                # Execute trades
                if signal == 'BUY' and not in_position:
    pass
    pass
    pass
                    # Calculate position size
                    position_size = self._calculate_position_size(capital, current_price, params)

                    if position_size > 0:
    pass
    pass
    pass
                        # Execute buy
                        entry_price = current_price * (1 + self.slippage)
                        entry_time = current_time
                        in_position = True

                        # Calculate leverage
                        leverage = self._calculate_leverage(params, current_price)

                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'position_size': position_size,
                            'leverage': leverage,
                            'type': 'LONG'
                        })

                elif signal == 'SELL' and in_position:
                    # Execute sell
                    exit_price = current_price * (1 - self.slippage)

                    # Calculate P&L
                    pnl = (exit_price - entry_price) / entry_price
                    trade_value = position_size * leverage
                    gross_pnl = trade_value * pnl
                    commission = trade_value * self.commission_rate * 2  # Entry + exit
                    net_pnl = gross_pnl - commission

                    # Update capital
                    capital += net_pnl

                    # Record trade
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'return': pnl
                    })

                    in_position = False
                    position_size = 0.0

                # Record equity
                equity_curve.append({
                    'timestamp': current_time,
                    'equity': capital,
                    'in_position': in_position
                })

            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'final_capital': capital,
                'total_return': (capital - self.initial_capital) / self.initial_capital
            }

        except Exception as e:
            self.logger.error(f"Backtesting error: {e}")
            return {'trades': [], 'equity_curve': [], 'final_capital': self.initial_capital, 'total_return': 0.0}

    def _generate_signal(self, row: pd.Series, params: Dict[str, Any], index: int) -> str:
    pass
    pass
    pass
        """Generate trading signal based on parameters and market data."""
        try:
            # Calculate technical indicators
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            min_idx = max(
                params.get('sma_fast_window', 20),
                params.get('sma_slow_window', 50),
                20
            )
            if index < min_idx:  # Need enough data
                return 'HOLD'

            # Parameterized moving averages
            sma_fast_window = int(params.get('sma_fast_window', 20))
            sma_slow_window = int(params.get('sma_slow_window', 50))
            sma_fast = self.market_data['close'].rolling(sma_fast_window).mean().iloc[index]
            sma_slow = self.market_data['close'].rolling(sma_slow_window).mean().iloc[index]

            # RSI thresholds
            rsi_oversold = float(params.get('rsi_oversold', 30.0))
            rsi_overbought = float(params.get('rsi_overbought', 70.0))
            rsi = self._calculate_rsi(self.market_data['close'].iloc[:index+1])

            # Volume ratio thresholds
            volume_ma_window = 20
            volume_ma = self.market_data['volume'].rolling(volume_ma_window).mean().iloc[index]
            volume_ratio = row['volume'] / volume_ma if volume_ma > 0 else 1.0
            vr_high = float(params.get('volume_ratio_high', 1.5))
            vr_low = float(params.get('volume_ratio_low', 0.5))

            # Strategy type selection
            strategy_type = params.get('strategy_type', 'trend_following')

            # Calculate confidence score
            confidence = 0.0

            if strategy_type == 'trend_following':
    pass
    pass
    pass
                # Trend: follow MA direction
                if sma_fast > sma_slow:
    pass
    pass
    pass
                    confidence += 0.3
                else:
                    confidence -= 0.3
                # RSI confirms momentum
                if rsi < rsi_oversold:
    pass
    pass
    pass
                    confidence += 0.2
                elif rsi > rsi_overbought:
                    confidence -= 0.2
            elif strategy_type == 'mean_reversion':
                # Mean reversion: prefer reversals
                if sma_fast < sma_slow and rsi < rsi_oversold:
    pass
    pass
    pass
                    confidence += 0.3
                if sma_fast > sma_slow and rsi > rsi_overbought:
    pass
    pass
    pass
                    confidence -= 0.3
            elif strategy_type == 'breakout':
                # Breakout: fast MA crossing slow + volume expansion
                if sma_fast > sma_slow and volume_ratio > vr_high:
    pass
    pass
    pass
                    confidence += 0.35
                elif sma_fast < sma_slow and volume_ratio > vr_high:
                    confidence -= 0.35

            # Volume analysis for all strategies
            if volume_ratio > vr_high:
    pass
    pass
    pass
                confidence += 0.1
            elif volume_ratio < vr_low:
                confidence -= 0.1

            # Apply confidence thresholds
            min_conf = float(params.get('min_confidence_threshold', 0.6))
            entry_th = float(params.get('entry_threshold', min_conf))
            threshold = max(min_conf, entry_th)

            if confidence >= threshold:
    pass
    pass
    pass
                return 'BUY'
            elif -confidence >= threshold:
                return 'SELL'

            return 'HOLD'

        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return 'HOLD'

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
    pass
    pass
    pass
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
    pass
                return 50.0

    except Exception as e:
        pass
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception:
            return 50.0

    def _calculate_position_size(self, capital: float, price: float, params: Dict[str, Any]) -> float:
    pass
    pass
    pass
        """Calculate position size based on Kelly criterion and ML confidence."""
        try:
            # Kelly criterion calculation (simplified)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            win_rate = 0.6  # Mock win rate
            avg_win = 0.02  # Mock average win
            avg_loss = 0.01  # Mock average loss

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp between 0 and 1

            # Apply Kelly multiplier
            kelly_multiplier = params.get('kelly_multiplier', 0.25)
            position_fraction = kelly_fraction * kelly_multiplier

            # Apply position size limits
            max_position_size = params.get('max_position_size', 0.5)
            min_position_size = params.get('min_position_size', 0.01)

            position_fraction = max(min_position_size, min(position_fraction, max_position_size))

            return position_fraction

        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return 0.01

    def _calculate_leverage(self, params: Dict[str, Any], price: float) -> float:
    pass
    pass
    pass
        """Calculate leverage based on parameters."""
        try:
            # Base leverage calculation
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            min_leverage = params.get('min_leverage', 10.0)
            max_leverage = params.get('max_leverage', 100.0)

            # Apply confidence-based adjustment
            confidence = 0.7  # Mock confidence score
            confidence_threshold = params.get('leverage_confidence_threshold', 0.6)

            if confidence >= confidence_threshold:
    pass
    pass
    pass
                leverage = min_leverage + (max_leverage - min_leverage) * (confidence - confidence_threshold) / (1 - confidence_threshold)
            else:
                leverage = min_leverage

            # Apply risk adjustment
            risk_adjustment = params.get('leverage_risk_adjustment', 1.0)
            leverage *= risk_adjustment

            # Apply maximum risk leverage limit
            max_risk_leverage = params.get('max_risk_leverage', 50.0)
            leverage = min(leverage, max_risk_leverage)

            return max(min_leverage, min(leverage, max_leverage))

        except Exception as e:
            self.logger.error(f"Leverage calculation error: {e}")
            return 10.0

    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
    pass
    pass
    pass
        """Calculate performance metrics from backtest results."""
        try:
            trades = backtest_results['trades']
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            equity_curve = backtest_results['equity_curve']
            final_capital = backtest_results['final_capital']

            if not trades:
    pass
    pass
    pass
                return {
                    'sharpe_ratio': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_return': 0.0
                }

            # Calculate returns
            equity_df = pd.DataFrame(equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)

            # Sharpe ratio
            if len(equity_df) > 1:
    pass
    pass
    pass
                sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252 * 24)  # Annualized
            else:
                sharpe_ratio = 0.0

            # Profit factor
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

            gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
            gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))

            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            profit_factor = min(profit_factor, 10.0)  # Cap at 10

            # Maximum drawdown
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = abs(equity_df['drawdown'].min())

            # Win rate
            win_rate = len(winning_trades) / len(trades) if trades else 0.0

            # Total return
            total_return = (final_capital - self.initial_capital) / self.initial_capital

            return {
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_return': total_return
            }

        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {e}")
            return {
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0
            }

    def _calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
    pass
    pass
    pass
        """Calculate weighted performance score."""
        try:
            score = 0.0

    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            # Sharpe ratio (higher is better)
            score += self.metric_weights['sharpe_ratio'] * min(metrics['sharpe_ratio'], 3.0) / 3.0

            # Profit factor (higher is better)
            score += self.metric_weights['profit_factor'] * min(metrics['profit_factor'], 5.0) / 5.0

            # Max drawdown (lower is better, so invert)
            score += self.metric_weights['max_drawdown'] * (1 - min(metrics['max_drawdown'], 0.5) / 0.5)

            # Win rate (higher is better)
            score += self.metric_weights['win_rate'] * metrics['win_rate']

            # Total return (higher is better)
            score += self.metric_weights['total_return'] * min(metrics['total_return'], 2.0) / 2.0

            return score

        except Exception as e:
            self.logger.error(f"Weighted score calculation error: {e}")
            return 0.0

    def get_detailed_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    pass
    pass
    pass
        """Get detailed analysis of backtest results."""
        try:
            trades = backtest_results['trades']
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            equity_curve = backtest_results['equity_curve']

            if not trades:
    pass
    pass
    pass
                return {"error": "No trades executed"}

            # Basic statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])

            # Trade analysis
            pnls = [t.get('pnl', 0) for t in trades]
            returns = [t.get('return', 0) for t in trades]

            analysis = {
                "summary": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                    "avg_trade_pnl": np.mean(pnls) if pnls else 0,
                    "avg_trade_return": np.mean(returns) if returns else 0,
                    "best_trade": max(pnls) if pnls else 0,
                    "worst_trade": min(pnls) if pnls else 0
                },
                "risk_metrics": {
                    "volatility": np.std(returns) if returns else 0,
                    "var_95": np.percentile(returns, 5) if returns else 0,
                    "max_consecutive_losses": self._calculate_max_consecutive_losses(trades)
                },
                "equity_analysis": {
                    "final_equity": equity_curve[-1]['equity'] if equity_curve else self.initial_capital,
                    "peak_equity": max(e['equity'] for e in equity_curve) if equity_curve else self.initial_capital,
                    "equity_volatility": np.std([e['equity'] for e in equity_curve]) if equity_curve else 0
                }
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Detailed analysis error: {e}")
            return {"error": str(e)}

    def _calculate_max_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
    pass
    pass
    pass
        """Calculate maximum consecutive losing trades."""
        try:
            max_consecutive = 0
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            current_consecutive = 0

            for trade in trades:
    pass
    pass
    pass
                if trade.get('pnl', 0) < 0:
    pass
    pass
    pass
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        except Exception:
            return 0


# Integration function for the parameter optimizer
async def evaluate_parameters_with_backtesting(params: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Evaluate parameters using backtesting (for integration with parameter optimizer).

    Args:
        params: Parameter dictionary
        config: Configuration dictionary

    Returns:
        float: Performance score
    """
    evaluator = BacktestingEvaluator(config)
    return await evaluator.evaluate_parameters(params)