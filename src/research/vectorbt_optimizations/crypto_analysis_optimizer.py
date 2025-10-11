"""
VectorBT Optimized Cryptocurrency Analysis

This module enhances the crypto analysis processor with VectorBT capabilities:
- Advanced technical indicators
- Optimized backtesting engine
- Enhanced performance metrics
- Portfolio analysis capabilities
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import warnings

# Suppress VectorBT warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)

class VectorBTCryptoOptimizer:
    """
    VectorBT-optimized cryptocurrency analysis processor.
    
    This class enhances the existing crypto analysis with VectorBT's capabilities:
    - Advanced technical indicators
    - Optimized backtesting
    - Enhanced performance metrics
    - Portfolio analysis
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize VectorBT crypto optimizer.
        
        Args:
            data_dir: Directory for data storage
            output_dir: Directory for results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # VectorBT configuration
        vbt.settings.set_theme("dark")
        vbt.settings['plotting']['layout']['width'] = 1200
        vbt.settings['plotting']['layout']['height'] = 600
        
        logger.info("✅ VectorBT crypto optimizer initialized")
    
    def enhance_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive technical indicators using VectorBT.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary of technical indicators
        """
        logger.info("🔧 Generating VectorBT technical indicators...")
        
        # Ensure data has proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Price data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        indicators = {}
        
        try:
            # Trend indicators
            indicators['sma_20'] = vbt.MA.run(close, 20).ma
            indicators['sma_50'] = vbt.MA.run(close, 50).ma
            indicators['ema_12'] = vbt.MA.run(close, 12, short_win=12).ma
            indicators['ema_26'] = vbt.MA.run(close, 26, short_win=26).ma
            
            # MACD
            macd = vbt.MACD.run(close)
            indicators['macd'] = macd.macd
            indicators['macd_signal'] = macd.signal
            indicators['macd_histogram'] = macd.histogram
            
            # RSI
            indicators['rsi'] = vbt.RSI.run(close).rsi
            
            # Bollinger Bands
            bb = vbt.BBANDS.run(close)
            indicators['bb_upper'] = bb.upper
            indicators['bb_middle'] = bb.middle
            indicators['bb_lower'] = bb.lower
            indicators['bb_width'] = bb.width
            indicators['bb_percent'] = bb.percent
            
            # Stochastic Oscillator
            stoch = vbt.STOCH.run(high, low, close)
            indicators['stoch_k'] = stoch.k
            indicators['stoch_d'] = stoch.d
            
            # Williams %R
            indicators['williams_r'] = vbt.WILLR.run(high, low, close).willr
            
            # ATR (Average True Range)
            indicators['atr'] = vbt.ATR.run(high, low, close).atr
            
            # ADX (Average Directional Index)
            adx = vbt.ADX.run(high, low, close)
            indicators['adx'] = adx.adx
            indicators['adx_pos'] = adx.plus_di
            indicators['adx_neg'] = adx.minus_di
            
            # Volume indicators
            indicators['obv'] = vbt.OBV.run(close, volume).obv
            indicators['ad'] = vbt.AD.run(high, low, close, volume).ad
            indicators['cmf'] = vbt.CMF.run(high, low, close, volume).cmf
            
            # Volatility indicators
            indicators['bb_width_norm'] = indicators['bb_width'] / close
            indicators['volatility'] = close.rolling(20).std()
            
            # Price patterns
            indicators['doji'] = vbt.DOJI.run(open=data['open'], high=high, low=low, close=close).doji
            indicators['hammer'] = vbt.HAMMER.run(open=data['open'], high=high, low=low, close=close).hammer
            indicators['shooting_star'] = vbt.SHOOTING_STAR.run(open=data['open'], high=high, low=low, close=close).shooting_star
            
            logger.info(f"✅ Generated {len(indicators)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}")
            return {}
        
        return indicators
    
    def create_trading_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Create trading signals using VectorBT indicators.
        
        Args:
            data: OHLCV data
            indicators: Technical indicators
            
        Returns:
            Dictionary of trading signals
        """
        logger.info("📊 Creating VectorBT trading signals...")
        
        close = data['close']
        signals = {}
        
        try:
            # Trend signals
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            if sma_20 is not None and sma_50 is not None:
                signals['trend_bullish'] = (sma_20 > sma_50).astype(int)
                signals['trend_bearish'] = (sma_20 < sma_50).astype(int)
            
            # MACD signals
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd is not None and macd_signal is not None:
                signals['macd_bullish'] = (macd > macd_signal).astype(int)
                signals['macd_bearish'] = (macd < macd_signal).astype(int)
            
            # RSI signals
            rsi = indicators.get('rsi')
            if rsi is not None:
                signals['rsi_oversold'] = (rsi < 30).astype(int)
                signals['rsi_overbought'] = (rsi > 70).astype(int)
                signals['rsi_bullish'] = ((rsi > 50) & (rsi.shift(1) <= 50)).astype(int)
                signals['rsi_bearish'] = ((rsi < 50) & (rsi.shift(1) >= 50)).astype(int)
            
            # Bollinger Bands signals
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            if bb_upper is not None and bb_lower is not None:
                signals['bb_breakout_upper'] = (close > bb_upper).astype(int)
                signals['bb_breakout_lower'] = (close < bb_lower).astype(int)
                signals['bb_squeeze'] = (indicators.get('bb_width', pd.Series()) < 
                                       indicators.get('bb_width', pd.Series()).rolling(20).mean()).astype(int)
            
            # Stochastic signals
            stoch_k = indicators.get('stoch_k')
            stoch_d = indicators.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                signals['stoch_oversold'] = ((stoch_k < 20) & (stoch_d < 20)).astype(int)
                signals['stoch_overbought'] = ((stoch_k > 80) & (stoch_d > 80)).astype(int)
                signals['stoch_bullish'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
                signals['stoch_bearish'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)
            
            # Volume signals
            obv = indicators.get('obv')
            if obv is not None:
                obv_sma = obv.rolling(20).mean()
                signals['volume_bullish'] = (obv > obv_sma).astype(int)
                signals['volume_bearish'] = (obv < obv_sma).astype(int)
            
            # Combined signals
            if len(signals) > 0:
                # Bullish consensus (multiple indicators agree)
                bullish_signals = [s for k, s in signals.items() if 'bullish' in k or 'oversold' in k]
                if bullish_signals:
                    signals['consensus_bullish'] = pd.concat(bullish_signals, axis=1).sum(axis=1) >= 2
                
                # Bearish consensus
                bearish_signals = [s for k, s in signals.items() if 'bearish' in k or 'overbought' in k]
                if bearish_signals:
                    signals['consensus_bearish'] = pd.concat(bearish_signals, axis=1).sum(axis=1) >= 2
            
            logger.info(f"✅ Created {len(signals)} trading signals")
            
        except Exception as e:
            logger.error(f"Error creating trading signals: {e}")
            return {}
        
        return signals
    
    def run_vectorbt_backtest(self, data: pd.DataFrame, signals: Dict[str, pd.Series], 
                            initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Run VectorBT backtesting on trading signals.
        
        Args:
            data: OHLCV data
            signals: Trading signals
            initial_capital: Initial capital for backtesting
            
        Returns:
            Backtesting results
        """
        logger.info("🚀 Running VectorBT backtesting...")
        
        close = data['close']
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
                    freq='1H'  # Assuming hourly data
                )
                
                # Extract performance metrics
                results[signal_name] = {
                    'total_return': pf.total_return(),
                    'sharpe_ratio': pf.sharpe_ratio(),
                    'max_drawdown': pf.max_drawdown(),
                    'win_rate': pf.trades.win_rate(),
                    'profit_factor': pf.trades.profit_factor(),
                    'total_trades': pf.trades.count(),
                    'avg_trade_duration': pf.trades.duration.mean(),
                    'portfolio_value': pf.value(),
                    'returns': pf.returns()
                }
            
            logger.info(f"✅ Completed backtesting for {len(results)} signals")
            
        except Exception as e:
            logger.error(f"Error in VectorBT backtesting: {e}")
            return {}
        
        return results
    
    def optimize_portfolio(self, data: pd.DataFrame, signals: Dict[str, pd.Series],
                          optimization_method: str = 'sharpe') -> Dict[str, Any]:
        """
        Optimize portfolio using VectorBT's portfolio optimization.
        
        Args:
            data: OHLCV data
            signals: Trading signals
            optimization_method: Optimization method ('sharpe', 'return', 'min_vol')
            
        Returns:
            Optimization results
        """
        logger.info(f"🎯 Running portfolio optimization ({optimization_method})...")
        
        close = data['close']
        results = {}
        
        try:
            # Prepare signal matrix
            signal_matrix = pd.DataFrame(signals).fillna(0)
            
            # Run portfolio optimization
            if optimization_method == 'sharpe':
                weights = vbt.Portfolio.from_signals(
                    close, 
                    entries=signal_matrix > 0,
                    exits=signal_matrix.shift(1) > 0
                ).optimize_sharpe_ratio()
            elif optimization_method == 'return':
                weights = vbt.Portfolio.from_signals(
                    close,
                    entries=signal_matrix > 0,
                    exits=signal_matrix.shift(1) > 0
                ).optimize_returns()
            else:  # min_vol
                weights = vbt.Portfolio.from_signals(
                    close,
                    entries=signal_matrix > 0,
                    exits=signal_matrix.shift(1) > 0
                ).optimize_min_volatility()
            
            # Calculate optimized portfolio metrics
            optimized_pf = vbt.Portfolio.from_signals(
                close,
                entries=signal_matrix > 0,
                exits=signal_matrix.shift(1) > 0,
                weights=weights
            )
            
            results = {
                'weights': weights,
                'total_return': optimized_pf.total_return(),
                'sharpe_ratio': optimized_pf.sharpe_ratio(),
                'max_drawdown': optimized_pf.max_drawdown(),
                'volatility': optimized_pf.returns().std(),
                'portfolio_value': optimized_pf.value()
            }
            
            logger.info(f"✅ Portfolio optimization completed")
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {}
        
        return results
    
    def generate_enhanced_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analysis using VectorBT.
        
        Args:
            data: OHLCV data
            
        Returns:
            Complete analysis results
        """
        logger.info("🔬 Generating VectorBT enhanced analysis...")
        
        # Generate technical indicators
        indicators = self.enhance_technical_indicators(data)
        
        # Create trading signals
        signals = self.create_trading_signals(data, indicators)
        
        # Run backtesting
        backtest_results = self.run_vectorbt_backtest(data, signals)
        
        # Optimize portfolio
        portfolio_results = self.optimize_portfolio(data, signals)
        
        # Compile results
        analysis_results = {
            'indicators': indicators,
            'signals': signals,
            'backtest_results': backtest_results,
            'portfolio_optimization': portfolio_results,
            'data_info': {
                'start_date': data.index.min(),
                'end_date': data.index.max(),
                'total_periods': len(data),
                'price_range': (data['close'].min(), data['close'].max()),
                'volatility': data['close'].pct_change().std()
            }
        }
        
        logger.info("✅ VectorBT enhanced analysis completed")
        return analysis_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "vectorbt_analysis.json"):
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in results.items():
            if key == 'indicators' or key == 'signals':
                # Convert pandas Series to dict
                serializable_results[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            elif key == 'backtest_results':
                # Convert backtest results
                serializable_results[key] = {
                    k: {
                        'total_return': float(v['total_return']) if not pd.isna(v['total_return']) else None,
                        'sharpe_ratio': float(v['sharpe_ratio']) if not pd.isna(v['sharpe_ratio']) else None,
                        'max_drawdown': float(v['max_drawdown']) if not pd.isna(v['max_drawdown']) else None,
                        'win_rate': float(v['win_rate']) if not pd.isna(v['win_rate']) else None,
                        'profit_factor': float(v['profit_factor']) if not pd.isna(v['profit_factor']) else None,
                        'total_trades': int(v['total_trades']) if not pd.isna(v['total_trades']) else 0
                    }
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        # Save to JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_path}")
    
    def create_visualization(self, data: pd.DataFrame, results: Dict[str, Any], 
                           save_path: Optional[str] = None):
        """
        Create comprehensive visualization using VectorBT.
        
        Args:
            data: OHLCV data
            results: Analysis results
            save_path: Path to save visualization
        """
        logger.info("📊 Creating VectorBT visualizations...")
        
        try:
            close = data['close']
            indicators = results.get('indicators', {})
            signals = results.get('signals', {})
            
            # Create subplots
            fig = vbt.make_subplots(
                rows=4, cols=1,
                subplot_titles=['Price & Signals', 'Technical Indicators', 'Volume', 'Performance'],
                vertical_spacing=0.05
            )
            
            # Price and signals
            fig.add_trace(
                vbt.plotting.Scatter(x=close.index, y=close.values, name='Close Price'),
                row=1, col=1
            )
            
            # Add some key indicators
            if 'sma_20' in indicators:
                fig.add_trace(
                    vbt.plotting.Scatter(x=indicators['sma_20'].index, y=indicators['sma_20'].values, name='SMA 20'),
                    row=1, col=1
                )
            
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                fig.add_trace(
                    vbt.plotting.Scatter(x=indicators['bb_upper'].index, y=indicators['bb_upper'].values, name='BB Upper'),
                    row=1, col=1
                )
                fig.add_trace(
                    vbt.plotting.Scatter(x=indicators['bb_lower'].index, y=indicators['bb_lower'].values, name='BB Lower'),
                    row=1, col=1
                )
            
            # RSI
            if 'rsi' in indicators:
                fig.add_trace(
                    vbt.plotting.Scatter(x=indicators['rsi'].index, y=indicators['rsi'].values, name='RSI'),
                    row=2, col=1
                )
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume
            fig.add_trace(
                vbt.plotting.Bar(x=data.index, y=data['volume'].values, name='Volume'),
                row=3, col=1
            )
            
            # Performance (if available)
            if 'consensus_bullish' in signals:
                portfolio_value = results.get('portfolio_optimization', {}).get('portfolio_value')
                if portfolio_value is not None:
                    fig.add_trace(
                        vbt.plotting.Scatter(x=portfolio_value.index, y=portfolio_value.values, name='Portfolio Value'),
                        row=4, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title="VectorBT Enhanced Cryptocurrency Analysis",
                height=1200,
                showlegend=True
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                logger.info(f"📊 Visualization saved to {save_path}")
            else:
                fig.show()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
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
    
    # Run VectorBT optimization
    optimizer = VectorBTCryptoOptimizer()
    results = optimizer.generate_enhanced_analysis(sample_data)
    
    # Save results
    optimizer.save_results(results)
    
    print("✅ VectorBT crypto analysis completed!")
    print(f"Generated {len(results['indicators'])} indicators")
    print(f"Created {len(results['signals'])} signals")
    print(f"Backtested {len(results['backtest_results'])} strategies")