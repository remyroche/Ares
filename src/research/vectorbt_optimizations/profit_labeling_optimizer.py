"""
VectorBT Optimized Profit Labeling Research

This module enhances the profit labeling research framework with VectorBT capabilities:
- VectorBT-based backtesting for profit labeling validation
- Enhanced performance metrics using VectorBT
- Signal-based profit target optimization
- Portfolio-level profit labeling analysis
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings

# Suppress VectorBT warnings
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)

class VectorBTProfitLabelingOptimizer:
    """
    VectorBT-optimized profit labeling research framework.
    
    This class enhances the existing profit labeling research with VectorBT capabilities:
    - VectorBT-based backtesting for validation
    - Enhanced performance metrics
    - Signal-based optimization
    - Portfolio-level analysis
    """
    
    def __init__(self, data: pd.DataFrame, profit_targets: Optional[Dict[str, float]] = None):
        """
        Initialize VectorBT profit labeling optimizer.
        
        Args:
            data: OHLCV data
            profit_targets: Dictionary of profit targets
        """
        self.data = data.copy()
        
        # Ensure proper index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Default profit targets
        self.profit_targets = profit_targets or {
            'micro': 0.002,   # 0.2%
            'small': 0.005,   # 0.5%
            'medium': 0.010,  # 1.0%
            'large': 0.020    # 2.0%
        }
        
        # VectorBT configuration
        vbt.settings.set_theme("dark")
        
        logger.info("✅ VectorBT profit labeling optimizer initialized")
    
    def generate_profit_signals(self, method: str = 'multi_horizon') -> Dict[str, pd.Series]:
        """
        Generate profit-based trading signals using VectorBT.
        
        Args:
            method: Signal generation method
            
        Returns:
            Dictionary of profit signals
        """
        logger.info(f"📊 Generating profit signals using method: {method}")
        
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        signals = {}
        
        try:
            if method == 'multi_horizon':
                signals = self._generate_multi_horizon_signals(close, high, low)
            elif method == 'momentum_based':
                signals = self._generate_momentum_signals(close, volume)
            elif method == 'volatility_based':
                signals = self._generate_volatility_signals(close, high, low)
            elif method == 'volume_based':
                signals = self._generate_volume_signals(close, volume)
            else:
                signals = self._generate_combined_signals(close, high, low, volume)
            
            logger.info(f"✅ Generated {len(signals)} profit signals")
            
        except Exception as e:
            logger.error(f"Error generating profit signals: {e}")
            return {}
        
        return signals
    
    def _generate_multi_horizon_signals(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Generate multi-horizon profit signals."""
        signals = {}
        
        # Calculate returns
        returns = close.pct_change()
        
        # Multi-horizon signals based on profit targets
        for target_name, target_value in self.profit_targets.items():
            # Immediate horizon (1-2 periods)
            immediate_horizon = 2
            future_returns = returns.shift(-immediate_horizon)
            signals[f'{target_name}_immediate'] = (future_returns >= target_value).astype(int)
            
            # Short horizon (3-5 periods)
            short_horizon = 5
            future_returns_short = returns.shift(-short_horizon)
            signals[f'{target_name}_short'] = (future_returns_short >= target_value).astype(int)
            
            # Medium horizon (6-10 periods)
            medium_horizon = 10
            future_returns_medium = returns.shift(-medium_horizon)
            signals[f'{target_name}_medium'] = (future_returns_medium >= target_value).astype(int)
        
        # Combined multi-horizon signal
        signal_cols = [col for col in signals.keys() if 'immediate' in col]
        if signal_cols:
            signal_df = pd.DataFrame({col: signals[col] for col in signal_cols})
            signals['multi_horizon_consensus'] = signal_df.sum(axis=1)
        
        return signals
    
    def _generate_momentum_signals(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Generate momentum-based profit signals."""
        signals = {}
        
        # Price momentum
        for period in [5, 10, 20]:
            momentum = close / close.shift(period) - 1
            signals[f'momentum_{period}'] = (momentum > 0).astype(int)
        
        # Volume momentum
        volume_momentum = volume / volume.rolling(20).mean()
        signals['volume_momentum'] = (volume_momentum > 1.2).astype(int)
        
        # RSI momentum
        rsi = vbt.RSI.run(close).rsi
        signals['rsi_momentum'] = ((rsi > 50) & (rsi.shift(1) <= 50)).astype(int)
        
        # MACD momentum
        macd = vbt.MACD.run(close)
        signals['macd_momentum'] = (macd.macd > macd.signal).astype(int)
        
        return signals
    
    def _generate_volatility_signals(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Generate volatility-based profit signals."""
        signals = {}
        
        # ATR-based signals
        atr = vbt.ATR.run(high, low, close).atr
        atr_ratio = atr / close
        
        # Low volatility (potential breakout)
        signals['low_volatility'] = (atr_ratio < atr_ratio.rolling(20).quantile(0.3)).astype(int)
        
        # High volatility (momentum continuation)
        signals['high_volatility'] = (atr_ratio > atr_ratio.rolling(20).quantile(0.7)).astype(int)
        
        # Bollinger Bands squeeze
        bb = vbt.BBANDS.run(close)
        bb_width = bb.width / close
        signals['bb_squeeze'] = (bb_width < bb_width.rolling(20).mean()).astype(int)
        
        # Volatility breakout
        signals['volatility_breakout'] = (atr_ratio > atr_ratio.rolling(20).mean() * 1.5).astype(int)
        
        return signals
    
    def _generate_volume_signals(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Generate volume-based profit signals."""
        signals = {}
        
        # Volume spikes
        volume_sma = volume.rolling(20).mean()
        signals['volume_spike'] = (volume > volume_sma * 2).astype(int)
        
        # OBV signals
        obv = vbt.OBV.run(close, volume).obv
        obv_sma = obv.rolling(20).mean()
        signals['obv_bullish'] = (obv > obv_sma).astype(int)
        
        # AD signals
        ad = vbt.AD.run(close, close, close, volume).ad  # Using close for high/low
        ad_sma = ad.rolling(20).mean()
        signals['ad_bullish'] = (ad > ad_sma).astype(int)
        
        # CMF signals
        cmf = vbt.CMF.run(close, close, close, volume).cmf
        signals['cmf_positive'] = (cmf > 0).astype(int)
        
        return signals
    
    def _generate_combined_signals(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Generate combined signals from all methods."""
        signals = {}
        
        # Combine all signal types
        momentum_signals = self._generate_momentum_signals(close, volume)
        volatility_signals = self._generate_volatility_signals(close, high, low)
        volume_signals = self._generate_volume_signals(close, volume)
        
        signals.update(momentum_signals)
        signals.update(volatility_signals)
        signals.update(volume_signals)
        
        # Create consensus signals
        signal_cols = list(signals.keys())
        if signal_cols:
            signal_df = pd.DataFrame({col: signals[col] for col in signal_cols})
            signals['consensus_bullish'] = (signal_df.sum(axis=1) >= len(signal_cols) // 2).astype(int)
            signals['strong_consensus'] = (signal_df.sum(axis=1) >= len(signal_cols) * 0.7).astype(int)
        
        return signals
    
    def backtest_profit_strategies(self, signals: Dict[str, pd.Series], 
                                 initial_capital: float = 10000) -> Dict[str, Dict[str, Any]]:
        """
        Backtest profit strategies using VectorBT.
        
        Args:
            signals: Trading signals
            initial_capital: Initial capital
            
        Returns:
            Backtesting results
        """
        logger.info("🚀 Backtesting profit strategies with VectorBT...")
        
        close = self.data['close']
        results = {}
        
        try:
            for signal_name, signal in signals.items():
                if signal.isna().all() or signal.sum() == 0:
                    continue
                
                # Create entries and exits
                entries = signal == 1
                exits = signal.shift(1) == 1  # Exit on next signal
                
                # Run backtest
                pf = vbt.Portfolio.from_signals(
                    close,
                    entries=entries,
                    exits=exits,
                    init_cash=initial_capital,
                    fees=0.001,  # 0.1% fees
                    freq='1H'
                )
                
                # Extract performance metrics
                results[signal_name] = {
                    'total_return': pf.total_return(),
                    'annualized_return': pf.annualized_return(),
                    'sharpe_ratio': pf.sharpe_ratio(),
                    'max_drawdown': pf.max_drawdown(),
                    'win_rate': pf.trades.win_rate(),
                    'profit_factor': pf.trades.profit_factor(),
                    'total_trades': pf.trades.count(),
                    'avg_trade_duration': pf.trades.duration.mean(),
                    'avg_win': pf.trades.winning.duration.mean() if pf.trades.winning.count() > 0 else 0,
                    'avg_loss': pf.trades.losing.duration.mean() if pf.trades.losing.count() > 0 else 0,
                    'portfolio_value': pf.value(),
                    'returns': pf.returns(),
                    'drawdowns': pf.drawdowns()
                }
            
            logger.info(f"✅ Backtested {len(results)} strategies")
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {}
        
        return results
    
    def optimize_profit_targets(self, method: str = 'sharpe') -> Dict[str, Any]:
        """
        Optimize profit targets using VectorBT.
        
        Args:
            method: Optimization method
            
        Returns:
            Optimization results
        """
        logger.info(f"🎯 Optimizing profit targets using {method}...")
        
        close = self.data['close']
        returns = close.pct_change()
        
        # Define parameter ranges
        target_values = np.arange(0.001, 0.05, 0.001)  # 0.1% to 5%
        horizon_values = [1, 2, 3, 5, 10, 20]
        
        best_score = -np.inf
        best_params = {}
        optimization_results = []
        
        try:
            for target in target_values:
                for horizon in horizon_values:
                    # Create signal
                    future_returns = returns.shift(-horizon)
                    signal = (future_returns >= target).astype(int)
                    
                    if signal.sum() == 0:
                        continue
                    
                    # Backtest
                    entries = signal == 1
                    exits = signal.shift(1) == 1
                    
                    pf = vbt.Portfolio.from_signals(
                        close,
                        entries=entries,
                        exits=exits,
                        init_cash=10000,
                        fees=0.001,
                        freq='1H'
                    )
                    
                    # Calculate score based on method
                    if method == 'sharpe':
                        score = pf.sharpe_ratio()
                    elif method == 'return':
                        score = pf.total_return()
                    elif method == 'win_rate':
                        score = pf.trades.win_rate()
                    else:  # composite
                        score = (
                            pf.sharpe_ratio() * 0.4 +
                            pf.total_return() * 0.3 +
                            pf.trades.win_rate() * 0.2 +
                            (1 - pf.max_drawdown()) * 0.1
                        )
                    
                    optimization_results.append({
                        'target': target,
                        'horizon': horizon,
                        'score': score,
                        'total_return': pf.total_return(),
                        'sharpe_ratio': pf.sharpe_ratio(),
                        'win_rate': pf.trades.win_rate(),
                        'max_drawdown': pf.max_drawdown(),
                        'total_trades': pf.trades.count()
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'target': target,
                            'horizon': horizon,
                            'score': score
                        }
            
            logger.info(f"✅ Optimization completed. Best score: {best_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return {}
        
        return {
            'best_params': best_params,
            'all_results': optimization_results,
            'optimization_method': method
        }
    
    def analyze_profit_consistency(self, signals: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Analyze profit consistency across different market conditions.
        
        Args:
            signals: Trading signals
            
        Returns:
            Consistency analysis results
        """
        logger.info("📊 Analyzing profit consistency...")
        
        close = self.data['close']
        returns = close.pct_change()
        
        # Define market conditions
        volatility = returns.rolling(20).std()
        trend = close.rolling(20).mean().pct_change()
        
        # Market regimes
        high_vol = volatility > volatility.quantile(0.7)
        low_vol = volatility < volatility.quantile(0.3)
        uptrend = trend > trend.quantile(0.7)
        downtrend = trend < trend.quantile(0.3)
        
        consistency_results = {}
        
        try:
            for signal_name, signal in signals.items():
                if signal.isna().all() or signal.sum() == 0:
                    continue
                
                # Calculate performance in different regimes
                regime_performance = {}
                
                for regime_name, regime_mask in [
                    ('high_volatility', high_vol),
                    ('low_volatility', low_vol),
                    ('uptrend', uptrend),
                    ('downtrend', downtrend)
                ]:
                    regime_returns = returns[regime_mask]
                    regime_signal = signal[regime_mask]
                    
                    if regime_signal.sum() == 0:
                        continue
                    
                    # Simple strategy performance
                    strategy_returns = regime_signal.shift(1) * regime_returns
                    
                    regime_performance[regime_name] = {
                        'mean_return': strategy_returns.mean(),
                        'std_return': strategy_returns.std(),
                        'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
                        'win_rate': (strategy_returns > 0).mean(),
                        'signal_count': regime_signal.sum()
                    }
                
                consistency_results[signal_name] = {
                    'regime_performance': regime_performance,
                    'consistency_score': self._calculate_consistency_score(regime_performance)
                }
            
            logger.info(f"✅ Analyzed consistency for {len(consistency_results)} signals")
            
        except Exception as e:
            logger.error(f"Error in consistency analysis: {e}")
            return {}
        
        return consistency_results
    
    def _calculate_consistency_score(self, regime_performance: Dict[str, Dict[str, float]]) -> float:
        """Calculate consistency score across regimes."""
        if not regime_performance:
            return 0.0
        
        sharpe_ratios = [perf['sharpe_ratio'] for perf in regime_performance.values()]
        if not sharpe_ratios:
            return 0.0
        
        # Consistency = 1 - coefficient of variation of Sharpe ratios
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        if mean_sharpe == 0:
            return 0.0
        
        return max(0, 1 - (std_sharpe / abs(mean_sharpe)))
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive VectorBT profit labeling analysis.
        
        Returns:
            Complete analysis results
        """
        logger.info("🔬 Running comprehensive VectorBT profit labeling analysis...")
        
        # Generate signals
        signals = self.generate_profit_signals('combined')
        
        # Backtest strategies
        backtest_results = self.backtest_profit_strategies(signals)
        
        # Optimize profit targets
        optimization_results = self.optimize_profit_targets('composite')
        
        # Analyze consistency
        consistency_results = self.analyze_profit_consistency(signals)
        
        # Generate summary
        summary = self._generate_analysis_summary(backtest_results, optimization_results, consistency_results)
        
        results = {
            'signals': signals,
            'backtest_results': backtest_results,
            'optimization_results': optimization_results,
            'consistency_results': consistency_results,
            'summary': summary,
            'data_info': {
                'start_date': self.data.index.min(),
                'end_date': self.data.index.max(),
                'total_periods': len(self.data),
                'price_range': (self.data['close'].min(), self.data['close'].max())
            }
        }
        
        logger.info("✅ Comprehensive VectorBT profit labeling analysis completed")
        return results
    
    def _generate_analysis_summary(self, backtest_results: Dict, optimization_results: Dict, 
                                 consistency_results: Dict) -> Dict[str, Any]:
        """Generate analysis summary."""
        summary = {
            'total_strategies': len(backtest_results),
            'profitable_strategies': len([r for r in backtest_results.values() if r['total_return'] > 0]),
            'best_strategy': None,
            'optimization_summary': {},
            'consistency_summary': {}
        }
        
        # Find best strategy
        if backtest_results:
            best_strategy = max(backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            summary['best_strategy'] = {
                'name': best_strategy[0],
                'sharpe_ratio': best_strategy[1]['sharpe_ratio'],
                'total_return': best_strategy[1]['total_return'],
                'win_rate': best_strategy[1]['win_rate']
            }
        
        # Optimization summary
        if optimization_results and 'best_params' in optimization_results:
            summary['optimization_summary'] = {
                'best_target': optimization_results['best_params']['target'],
                'best_horizon': optimization_results['best_params']['horizon'],
                'best_score': optimization_results['best_params']['score']
            }
        
        # Consistency summary
        if consistency_results:
            avg_consistency = np.mean([r['consistency_score'] for r in consistency_results.values()])
            summary['consistency_summary'] = {
                'average_consistency': avg_consistency,
                'high_consistency_strategies': len([r for r in consistency_results.values() if r['consistency_score'] > 0.7])
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = "vectorbt_profit_labeling.json"):
        """Save analysis results to file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if key == 'signals':
                serializable_results[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            elif key == 'backtest_results':
                serializable_results[key] = {
                    k: {
                        'total_return': float(v['total_return']) if not pd.isna(v['total_return']) else None,
                        'sharpe_ratio': float(v['sharpe_ratio']) if not pd.isna(v['sharpe_ratio']) else None,
                        'max_drawdown': float(v['max_drawdown']) if not pd.isna(v['max_drawdown']) else None,
                        'win_rate': float(v['win_rate']) if not pd.isna(v['win_rate']) else None,
                        'total_trades': int(v['total_trades']) if not pd.isna(v['total_trades']) else 0
                    }
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        import json
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        sample_data.loc[sample_data.index[i], 'high'] = max(sample_data.iloc[i][['open', 'high', 'low', 'close']])
        sample_data.loc[sample_data.index[i], 'low'] = min(sample_data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Run VectorBT profit labeling analysis
    optimizer = VectorBTProfitLabelingOptimizer(sample_data)
    results = optimizer.run_comprehensive_analysis()
    
    # Save results
    optimizer.save_results(results)
    
    print("✅ VectorBT profit labeling analysis completed!")
    print(f"Generated {len(results['signals'])} signals")
    print(f"Backtested {len(results['backtest_results'])} strategies")
    print(f"Best strategy: {results['summary']['best_strategy']}")