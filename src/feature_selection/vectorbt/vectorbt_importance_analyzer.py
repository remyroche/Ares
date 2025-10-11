"""
VectorBT Enhanced Financial Feature Importance Analyzer

This module provides comprehensive financial feature importance analysis
using VectorBT's advanced financial metrics and technical indicators.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Import VectorBT with fallback
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available. Some financial metrics will be disabled.")

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error, tprint_performance
from src.utils.dependency_manager import DependencyManager

logger = logging.getLogger(__name__)

@dataclass
class VectorBTImportanceConfig:
    """Configuration for VectorBT financial feature importance analysis."""
    # Financial metrics to include
    include_technical_indicators: bool = True
    include_risk_metrics: bool = True
    include_performance_metrics: bool = True
    include_microstructure_metrics: bool = True
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # Risk metrics parameters
    sharpe_lookback: int = 252
    sortino_lookback: int = 252
    var_confidence: float = 0.05
    max_drawdown_lookback: int = 252
    
    # Performance metrics parameters
    return_lookback: int = 252
    volatility_lookback: int = 252
    
    # Microstructure parameters
    bid_ask_spread_threshold: float = 0.001
    volume_threshold: float = 1000
    
    # General settings
    enable_parallel: bool = True
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True

@dataclass
class FinancialImportanceResult:
    """Result of financial feature importance analysis."""
    feature_names: List[str]
    technical_indicators: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, Dict[str, float]]
    performance_metrics: Dict[str, Dict[str, float]]
    microstructure_metrics: Dict[str, Dict[str, float]]
    combined_scores: Dict[str, float]
    market_regime: str
    analysis_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class VectorBTImportanceAnalyzer:
    """Enhanced financial feature importance analyzer using VectorBT."""
    
    def __init__(self, config: Optional[VectorBTImportanceConfig] = None):
        """Initialize VectorBT importance analyzer."""
        self.config = config or VectorBTImportanceConfig()
        self.logger = logger.getChild('VectorBTImportanceAnalyzer')
        self.dependency_manager = DependencyManager()
        
        # Check VectorBT availability
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available. Using fallback implementations.")
            self.vectorbt_available = False
        else:
            self.vectorbt_available = True
            tprint_success("✅ VectorBT available for enhanced financial analysis")
        
        # Performance tracking
        self.performance_stats = {
            'analyses_performed': 0,
            'total_time': 0.0,
            'technical_indicators_calculated': 0,
            'risk_metrics_calculated': 0
        }
        
        tprint_success("🚀 VectorBTImportanceAnalyzer initialized")
    
    def _ensure_vectorbt_data(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Ensure data is in VectorBT-compatible format."""
        if isinstance(data, np.ndarray):
            # Convert to DataFrame with proper column names
            if data.ndim == 1:
                return pd.DataFrame({'price': data})
            else:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
                return pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise ValueError("Data must be numpy array or pandas DataFrame")
    
    def _calculate_technical_indicators(self, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate technical indicators using VectorBT."""
        if not self.vectorbt_available:
            return self._calculate_technical_indicators_fallback(prices)
        
        try:
            indicators = {}
            
            # Ensure we have price data
            if 'price' not in prices.columns and len(prices.columns) > 0:
                price_col = prices.columns[0]  # Use first column as price
            else:
                price_col = 'price'
            
            price_series = prices[price_col] if price_col in prices.columns else prices.iloc[:, 0]
            
            # RSI
            if self.config.include_technical_indicators:
                rsi = vbt.RSI.run(price_series, window=self.config.rsi_period).rsi
                indicators['rsi'] = {
                    'current': float(rsi.iloc[-1]) if not rsi.empty else 0.0,
                    'mean': float(rsi.mean()) if not rsi.empty else 0.0,
                    'std': float(rsi.std()) if not rsi.empty else 0.0,
                    'trend': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral'
                }
                
                # MACD
                macd = vbt.MACD.run(price_series, 
                                  fast_window=self.config.macd_fast,
                                  slow_window=self.config.macd_slow,
                                  signal_window=self.config.macd_signal)
                indicators['macd'] = {
                    'macd': float(macd.macd.iloc[-1]) if not macd.macd.empty else 0.0,
                    'signal': float(macd.signal.iloc[-1]) if not macd.signal.empty else 0.0,
                    'histogram': float(macd.histogram.iloc[-1]) if not macd.histogram.empty else 0.0,
                    'crossover': 'bullish' if macd.macd.iloc[-1] > macd.signal.iloc[-1] else 'bearish'
                }
                
                # Bollinger Bands
                bb = vbt.BBANDS.run(price_series, 
                                   window=self.config.bb_period,
                                   alpha=self.config.bb_std)
                indicators['bollinger_bands'] = {
                    'upper': float(bb.upper.iloc[-1]) if not bb.upper.empty else 0.0,
                    'middle': float(bb.middle.iloc[-1]) if not bb.middle.empty else 0.0,
                    'lower': float(bb.lower.iloc[-1]) if not bb.lower.empty else 0.0,
                    'width': float((bb.upper.iloc[-1] - bb.lower.iloc[-1]) / bb.middle.iloc[-1]) if not bb.middle.empty else 0.0,
                    'position': 'above' if price_series.iloc[-1] > bb.upper.iloc[-1] else 'below' if price_series.iloc[-1] < bb.lower.iloc[-1] else 'within'
                }
                
                # Stochastic Oscillator
                stoch = vbt.STOCH.run(price_series, 
                                     k_window=self.config.stoch_k_period,
                                     d_window=self.config.stoch_d_period)
                indicators['stochastic'] = {
                    'k_percent': float(stoch.k_percent.iloc[-1]) if not stoch.k_percent.empty else 0.0,
                    'd_percent': float(stoch.d_percent.iloc[-1]) if not stoch.d_percent.empty else 0.0,
                    'overbought': stoch.k_percent.iloc[-1] > 80 if not stoch.k_percent.empty else False,
                    'oversold': stoch.k_percent.iloc[-1] < 20 if not stoch.k_percent.empty else False
                }
            
            self.performance_stats['technical_indicators_calculated'] += len(indicators)
            return indicators
            
        except Exception as e:
            self.logger.warning(f"Technical indicators calculation failed: {e}")
            return self._calculate_technical_indicators_fallback(prices)
    
    def _calculate_technical_indicators_fallback(self, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Fallback technical indicators calculation without VectorBT."""
        indicators = {}
        
        if len(prices) < 2:
            return indicators
        
        price_series = prices.iloc[:, 0] if len(prices.columns) > 0 else prices.squeeze()
        
        # Simple RSI calculation
        if len(price_series) >= self.config.rsi_period + 1:
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            indicators['rsi'] = {
                'current': float(rsi.iloc[-1]) if not rsi.empty else 0.0,
                'mean': float(rsi.mean()) if not rsi.empty else 0.0,
                'std': float(rsi.std()) if not rsi.empty else 0.0,
                'trend': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral'
            }
        
        return indicators
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate risk metrics using VectorBT."""
        if not self.vectorbt_available or len(returns) < 2:
            return self._calculate_risk_metrics_fallback(returns)
        
        try:
            risk_metrics = {}
            
            # Ensure returns is a pandas Series
            if isinstance(returns, np.ndarray):
                returns = pd.Series(returns)
            
            # Sharpe Ratio
            if self.config.include_risk_metrics:
                sharpe_ratio = vbt.sharpe_ratio(returns, 
                                              window=self.config.sharpe_lookback,
                                              risk_free=0.0)
                risk_metrics['sharpe_ratio'] = {
                    'current': float(sharpe_ratio.iloc[-1]) if not sharpe_ratio.empty else 0.0,
                    'mean': float(sharpe_ratio.mean()) if not sharpe_ratio.empty else 0.0,
                    'std': float(sharpe_ratio.std()) if not sharpe_ratio.empty else 0.0
                }
                
                # Sortino Ratio
                sortino_ratio = vbt.sortino_ratio(returns,
                                                window=self.config.sortino_lookback,
                                                risk_free=0.0)
                risk_metrics['sortino_ratio'] = {
                    'current': float(sortino_ratio.iloc[-1]) if not sortino_ratio.empty else 0.0,
                    'mean': float(sortino_ratio.mean()) if not sortino_ratio.empty else 0.0,
                    'std': float(sortino_ratio.std()) if not sortino_ratio.empty else 0.0
                }
                
                # Maximum Drawdown
                max_dd = vbt.max_drawdown(returns, 
                                        window=self.config.max_drawdown_lookback)
                risk_metrics['max_drawdown'] = {
                    'current': float(max_dd.iloc[-1]) if not max_dd.empty else 0.0,
                    'mean': float(max_dd.mean()) if not max_dd.empty else 0.0,
                    'worst': float(max_dd.min()) if not max_dd.empty else 0.0
                }
                
                # Value at Risk (VaR)
                var = vbt.var(returns, 
                            window=self.config.sharpe_lookback,
                            alpha=self.config.var_confidence)
                risk_metrics['var'] = {
                    'current': float(var.iloc[-1]) if not var.empty else 0.0,
                    'mean': float(var.mean()) if not var.empty else 0.0,
                    'std': float(var.std()) if not var.empty else 0.0
                }
            
            self.performance_stats['risk_metrics_calculated'] += len(risk_metrics)
            return risk_metrics
            
        except Exception as e:
            self.logger.warning(f"Risk metrics calculation failed: {e}")
            return self._calculate_risk_metrics_fallback(returns)
    
    def _calculate_risk_metrics_fallback(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Fallback risk metrics calculation without VectorBT."""
        risk_metrics = {}
        
        if len(returns) < 2:
            return risk_metrics
        
        # Simple Sharpe Ratio
        if len(returns) >= self.config.sharpe_lookback:
            mean_return = returns.rolling(window=self.config.sharpe_lookback).mean()
            std_return = returns.rolling(window=self.config.sharpe_lookback).std()
            sharpe = mean_return / std_return
            
            risk_metrics['sharpe_ratio'] = {
                'current': float(sharpe.iloc[-1]) if not sharpe.empty else 0.0,
                'mean': float(sharpe.mean()) if not sharpe.empty else 0.0,
                'std': float(sharpe.std()) if not sharpe.empty else 0.0
            }
        
        return risk_metrics
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics using VectorBT."""
        if not self.vectorbt_available or len(returns) < 2:
            return self._calculate_performance_metrics_fallback(returns)
        
        try:
            performance_metrics = {}
            
            if isinstance(returns, np.ndarray):
                returns = pd.Series(returns)
            
            # Cumulative Returns
            if self.config.include_performance_metrics:
                cum_returns = vbt.cumulative_returns(returns)
                performance_metrics['cumulative_returns'] = {
                    'total': float(cum_returns.iloc[-1]) if not cum_returns.empty else 0.0,
                    'annualized': float(cum_returns.iloc[-1] ** (252 / len(returns))) if not cum_returns.empty else 0.0
                }
                
                # Rolling Returns
                rolling_returns = returns.rolling(window=self.config.return_lookback).mean()
                performance_metrics['rolling_returns'] = {
                    'current': float(rolling_returns.iloc[-1]) if not rolling_returns.empty else 0.0,
                    'mean': float(rolling_returns.mean()) if not rolling_returns.empty else 0.0,
                    'std': float(rolling_returns.std()) if not rolling_returns.empty else 0.0
                }
                
                # Volatility
                volatility = returns.rolling(window=self.config.volatility_lookback).std() * np.sqrt(252)
                performance_metrics['volatility'] = {
                    'current': float(volatility.iloc[-1]) if not volatility.empty else 0.0,
                    'mean': float(volatility.mean()) if not volatility.empty else 0.0,
                    'std': float(volatility.std()) if not volatility.empty else 0.0
                }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return self._calculate_performance_metrics_fallback(returns)
    
    def _calculate_performance_metrics_fallback(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Fallback performance metrics calculation without VectorBT."""
        performance_metrics = {}
        
        if len(returns) < 2:
            return performance_metrics
        
        # Simple cumulative returns
        cum_returns = (1 + returns).cumprod() - 1
        performance_metrics['cumulative_returns'] = {
            'total': float(cum_returns.iloc[-1]) if not cum_returns.empty else 0.0,
            'annualized': float(cum_returns.iloc[-1] ** (252 / len(returns))) if not cum_returns.empty else 0.0
        }
        
        return performance_metrics
    
    def _detect_market_regime(self, prices: pd.DataFrame, returns: pd.Series) -> str:
        """Detect current market regime using VectorBT."""
        if not self.vectorbt_available or len(returns) < 50:
            return 'unknown'
        
        try:
            # Calculate volatility regime
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            current_vol = volatility.iloc[-1] if not volatility.empty else 0.0
            avg_vol = volatility.mean() if not volatility.empty else 0.0
            
            # Calculate trend regime
            if len(prices) > 0:
                price_series = prices.iloc[:, 0] if len(prices.columns) > 0 else prices.squeeze()
                sma_short = price_series.rolling(window=10).mean()
                sma_long = price_series.rolling(window=50).mean()
                
                if not sma_short.empty and not sma_long.empty:
                    trend = 'uptrend' if sma_short.iloc[-1] > sma_long.iloc[-1] else 'downtrend'
                else:
                    trend = 'sideways'
            else:
                trend = 'sideways'
            
            # Combine regimes
            if current_vol > avg_vol * 1.5:
                vol_regime = 'high_volatility'
            elif current_vol < avg_vol * 0.5:
                vol_regime = 'low_volatility'
            else:
                vol_regime = 'normal_volatility'
            
            return f"{trend}_{vol_regime}"
            
        except Exception as e:
            self.logger.warning(f"Market regime detection failed: {e}")
            return 'unknown'
    
    def analyze_financial_importance(self, 
                                   prices: Union[np.ndarray, pd.DataFrame],
                                   returns: Union[np.ndarray, pd.Series],
                                   feature_names: Optional[List[str]] = None) -> FinancialImportanceResult:
        """Analyze financial feature importance using VectorBT metrics."""
        tprint("🔍 Starting VectorBT financial importance analysis")
        
        start_time = datetime.now()
        
        try:
            # Prepare data
            prices_df = self._ensure_vectorbt_data(prices)
            if isinstance(returns, np.ndarray):
                returns_series = pd.Series(returns)
            else:
                returns_series = returns
            
            # Generate feature names if not provided
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(prices_df.shape[1])]
            
            # Calculate technical indicators
            technical_indicators = {}
            if self.config.include_technical_indicators:
                technical_indicators = self._calculate_technical_indicators(prices_df)
            
            # Calculate risk metrics
            risk_metrics = {}
            if self.config.include_risk_metrics:
                risk_metrics = self._calculate_risk_metrics(returns_series)
            
            # Calculate performance metrics
            performance_metrics = {}
            if self.config.include_performance_metrics:
                performance_metrics = self._calculate_performance_metrics(returns_series)
            
            # Calculate microstructure metrics (simplified)
            microstructure_metrics = {}
            if self.config.include_microstructure_metrics:
                microstructure_metrics = self._calculate_microstructure_metrics(prices_df, returns_series)
            
            # Detect market regime
            market_regime = self._detect_market_regime(prices_df, returns_series)
            
            # Calculate combined scores
            combined_scores = self._calculate_combined_scores(
                technical_indicators, risk_metrics, performance_metrics, microstructure_metrics
            )
            
            # Create result
            result = FinancialImportanceResult(
                feature_names=feature_names,
                technical_indicators=technical_indicators,
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics,
                microstructure_metrics=microstructure_metrics,
                combined_scores=combined_scores,
                market_regime=market_regime,
                analysis_metadata={
                    'vectorbt_available': self.vectorbt_available,
                    'data_length': len(prices_df),
                    'analysis_time': (datetime.now() - start_time).total_seconds(),
                    'config': self.config.__dict__
                }
            )
            
            # Update performance stats
            self.performance_stats['analyses_performed'] += 1
            self.performance_stats['total_time'] += (datetime.now() - start_time).total_seconds()
            
            tprint_success(f"✅ Financial importance analysis completed: {market_regime} regime")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Financial importance analysis failed: {e}")
            tprint_error(f"❌ Analysis failed: {e}")
            raise
    
    def _calculate_microstructure_metrics(self, prices: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate microstructure metrics (simplified implementation)."""
        microstructure_metrics = {}
        
        # Simplified bid-ask spread proxy (using price volatility)
        if len(prices) > 1:
            price_volatility = prices.iloc[:, 0].pct_change().std() if len(prices.columns) > 0 else prices.squeeze().pct_change().std()
            microstructure_metrics['bid_ask_spread_proxy'] = {
                'current': float(price_volatility),
                'mean': float(price_volatility),
                'std': 0.0
            }
        
        return microstructure_metrics
    
    def _calculate_combined_scores(self, 
                                 technical_indicators: Dict,
                                 risk_metrics: Dict,
                                 performance_metrics: Dict,
                                 microstructure_metrics: Dict) -> Dict[str, float]:
        """Calculate combined importance scores."""
        combined_scores = {}
        
        # Weight different metric types
        weights = {
            'technical': 0.3,
            'risk': 0.3,
            'performance': 0.3,
            'microstructure': 0.1
        }
        
        # Extract scores from each category
        for category, weight in weights.items():
            if category == 'technical' and technical_indicators:
                for indicator, metrics in technical_indicators.items():
                    if 'current' in metrics:
                        combined_scores[f"{indicator}_importance"] = metrics['current'] * weight
            
            elif category == 'risk' and risk_metrics:
                for metric, values in risk_metrics.items():
                    if 'current' in values:
                        combined_scores[f"{metric}_importance"] = abs(values['current']) * weight
            
            elif category == 'performance' and performance_metrics:
                for metric, values in performance_metrics.items():
                    if 'current' in values:
                        combined_scores[f"{metric}_importance"] = abs(values['current']) * weight
            
            elif category == 'microstructure' and microstructure_metrics:
                for metric, values in microstructure_metrics.items():
                    if 'current' in values:
                        combined_scores[f"{metric}_importance"] = values['current'] * weight
        
        return combined_scores
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['analyses_performed'] > 0:
            stats['avg_time_per_analysis'] = stats['total_time'] / stats['analyses_performed']
        else:
            stats['avg_time_per_analysis'] = 0.0
        
        tprint_performance(f"📊 VectorBT Importance Stats: {stats['analyses_performed']} analyses, "
                         f"{stats['avg_time_per_analysis']:.3f}s avg")
        
        return stats

def create_vectorbt_importance_analyzer(config: Optional[VectorBTImportanceConfig] = None) -> VectorBTImportanceAnalyzer:
    """Create a VectorBT importance analyzer."""
    return VectorBTImportanceAnalyzer(config)