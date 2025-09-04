"""
Performance Attribution System for 20 HMM Clusters and Timeframes
Tracks performance across clusters, timeframes, and barrier configurations
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.core.decorators.errors import handles_errors


@dataclass
class TradeAttribution:
    """Attribution data for a single trade"""
    trade_id: str
    timestamp: datetime
    regime: str
    timeframe: str
    barrier_type: str
    pnl: float
    confidence: float
    leverage: float
    execution_time: float
    metadata: Dict[str, Any]


class PerformanceAttributionSystem:
    """
    Performance Attribution System for high-leverage trading.
    Tracks performance across 20 HMM clusters and 15m-30m-1h timeframes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Performance Attribution System.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('PerformanceAttribution')
        
        # Configuration
        self.attribution_config = config.get('performance_attribution', {})
        self.regime_names = [f"regime_{i:02d}" for i in range(20)]  # regime_00 to regime_19
        self.timeframes = ['5m', '15m', '30m', '1h']  # Added 5m for high-leverage trading
        self.leverage_multiplier = self.attribution_config.get('leverage_multiplier', 10)
        
        # Storage
        self.trade_attributions: deque = deque(maxlen=10000)  # Last 10,000 trades
        self.regime_metrics: Dict[str, Dict[str, Any]] = {}
        self.timeframe_metrics: Dict[str, Dict[str, Any]] = {}
        self.combined_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.rolling_windows = {
            'short': 100,    # Last 100 trades
            'medium': 500,   # Last 500 trades
            'long': 1000     # Last 1000 trades
        }
        
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='performance attribution initialization')
    async def initialize(self) -> bool:
        """
        Initialize the Performance Attribution System.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Performance Attribution System...")
            
            # Initialize metrics storage
            for regime in self.regime_names:
                self.regime_metrics[regime] = {
                    'trades': deque(maxlen=self.rolling_windows['long']),
                    'total_pnl': 0.0,
                    'trade_count': 0,
                    'last_updated': datetime.now()
                }
            
            for timeframe in self.timeframes:
                self.timeframe_metrics[timeframe] = {
                    'trades': deque(maxlen=self.rolling_windows['long']),
                    'total_pnl': 0.0,
                    'trade_count': 0,
                    'last_updated': datetime.now()
                }
            
            # Initialize combined metrics
            for regime in self.regime_names:
                for timeframe in self.timeframes:
                    key = f"{regime}_{timeframe}"
                    self.combined_metrics[key] = {
                        'trades': deque(maxlen=self.rolling_windows['long']),
                        'total_pnl': 0.0,
                        'trade_count': 0,
                        'last_updated': datetime.now()
                    }
            
            self.logger.info("✅ Performance Attribution System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Performance Attribution initialization failed: {e}")
            return False
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='trade attribution recording')
    async def record_trade_attribution(
        self,
        trade_id: str,
        regime: str,
        timeframe: str,
        barrier_type: str,
        pnl: float,
        confidence: float,
        leverage: float = None,
        execution_time: float = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[TradeAttribution]:
        """
        Record trade attribution for performance tracking.
        
        Args:
            trade_id: Unique trade identifier
            regime: HMM regime name
            timeframe: Trading timeframe
            barrier_type: Type of barrier used
            pnl: Profit and loss
            confidence: Trade confidence
            leverage: Leverage used (defaults to system leverage)
            execution_time: Trade execution time in seconds
            metadata: Additional trade metadata
            
        Returns:
            TradeAttribution: Recorded attribution data
        """
        try:
            # Validate inputs
            if regime not in self.regime_names:
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            if timeframe not in self.timeframes:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Create attribution record
            attribution = TradeAttribution(
                trade_id=trade_id,
                timestamp=datetime.now(),
                regime=regime,
                timeframe=timeframe,
                barrier_type=barrier_type,
                pnl=pnl * (leverage or self.leverage_multiplier),  # Apply leverage
                confidence=confidence,
                leverage=leverage or self.leverage_multiplier,
                execution_time=execution_time or 0.0,
                metadata=metadata or {}
            )
            
            # Store attribution
            self.trade_attributions.append(attribution)
            
            # Update metrics
            await self._update_attribution_metrics(attribution)
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error recording trade attribution: {e}")
            return None
    
    async def _update_attribution_metrics(self, attribution: TradeAttribution) -> None:
        """Update attribution metrics with new trade"""
        
        # Update regime metrics
        regime_metrics = self.regime_metrics[attribution.regime]
        regime_metrics['trades'].append(attribution.pnl)
        regime_metrics['total_pnl'] += attribution.pnl
        regime_metrics['trade_count'] += 1
        regime_metrics['last_updated'] = attribution.timestamp
        
        # Update timeframe metrics
        timeframe_metrics = self.timeframe_metrics[attribution.timeframe]
        timeframe_metrics['trades'].append(attribution.pnl)
        timeframe_metrics['total_pnl'] += attribution.pnl
        timeframe_metrics['trade_count'] += 1
        timeframe_metrics['last_updated'] = attribution.timestamp
        
        # Update combined metrics
        combined_key = f"{attribution.regime}_{attribution.timeframe}"
        combined_metrics = self.combined_metrics[combined_key]
        combined_metrics['trades'].append(attribution.pnl)
        combined_metrics['total_pnl'] += attribution.pnl
        combined_metrics['trade_count'] += 1
        combined_metrics['last_updated'] = attribution.timestamp
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='regime performance analysis')
    async def analyze_regime_performance(
        self,
        regime: str = None,
        window: str = 'medium'
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze performance for specific regime or all regimes.
        
        Args:
            regime: Specific regime to analyze (None for all regimes)
            window: Analysis window ('short', 'medium', 'long')
            
        Returns:
            Dict: Performance analysis results
        """
        try:
            if window not in self.rolling_windows:
                self.logger.error(f"Invalid window: {window}")
                return None
            
            if regime:
                # Analyze specific regime
                if regime not in self.regime_metrics:
                    self.logger.error(f"Regime {regime} not found")
                    return None
                
                return self._analyze_single_regime(regime, window)
            else:
                # Analyze all regimes
                return self._analyze_all_regimes(window)
                
        except Exception as e:
            self.logger.error(f"Error analyzing regime performance: {e}")
            return None
    
    def _analyze_single_regime(self, regime: str, window: str) -> Dict[str, Any]:
        """Analyze performance for a single regime"""
        
        regime_metrics = self.regime_metrics[regime]
        trades = list(regime_metrics['trades'])
        
        if not trades:
            return {
                'regime': regime,
                'status': 'no_data',
                'trade_count': 0
            }
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(trades, window)
        
        return {
            'regime': regime,
            'status': 'analyzed',
            'window': window,
            'trade_count': len(trades),
            'performance_metrics': performance_metrics,
            'last_updated': regime_metrics['last_updated']
        }
    
    def _analyze_all_regimes(self, window: str) -> Dict[str, Any]:
        """Analyze performance for all regimes"""
        
        regime_analyses = {}
        overall_metrics = {
            'total_trades': 0,
            'total_pnl': 0.0,
            'regime_count': 0,
            'best_regime': None,
            'worst_regime': None
        }
        
        regime_performances = []
        
        for regime in self.regime_names:
            regime_analysis = self._analyze_single_regime(regime, window)
            regime_analyses[regime] = regime_analysis
            
            if regime_analysis['status'] == 'analyzed':
                overall_metrics['total_trades'] += regime_analysis['trade_count']
                overall_metrics['total_pnl'] += regime_analysis['performance_metrics']['total_pnl']
                overall_metrics['regime_count'] += 1
                
                regime_performances.append({
                    'regime': regime,
                    'total_pnl': regime_analysis['performance_metrics']['total_pnl'],
                    'sharpe_ratio': regime_analysis['performance_metrics']['sharpe_ratio']
                })
        
        # Find best and worst regimes
        if regime_performances:
            best_regime = max(regime_performances, key=lambda x: x['total_pnl'])
            worst_regime = min(regime_performances, key=lambda x: x['total_pnl'])
            
            overall_metrics['best_regime'] = best_regime['regime']
            overall_metrics['worst_regime'] = worst_regime['regime']
        
        return {
            'analysis_timestamp': datetime.now(),
            'window': window,
            'regime_analyses': regime_analyses,
            'overall_metrics': overall_metrics,
            'regime_rankings': sorted(regime_performances, key=lambda x: x['total_pnl'], reverse=True)
        }
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='timeframe performance analysis')
    async def analyze_timeframe_performance(
        self,
        timeframe: str = None,
        window: str = 'medium'
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze performance for specific timeframe or all timeframes.
        
        Args:
            timeframe: Specific timeframe to analyze (None for all timeframes)
            window: Analysis window ('short', 'medium', 'long')
            
        Returns:
            Dict: Performance analysis results
        """
        try:
            if window not in self.rolling_windows:
                self.logger.error(f"Invalid window: {window}")
                return None
            
            if timeframe:
                # Analyze specific timeframe
                if timeframe not in self.timeframe_metrics:
                    self.logger.error(f"Timeframe {timeframe} not found")
                    return None
                
                return self._analyze_single_timeframe(timeframe, window)
            else:
                # Analyze all timeframes
                return self._analyze_all_timeframes(window)
                
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe performance: {e}")
            return None
    
    def _analyze_single_timeframe(self, timeframe: str, window: str) -> Dict[str, Any]:
        """Analyze performance for a single timeframe"""
        
        timeframe_metrics = self.timeframe_metrics[timeframe]
        trades = list(timeframe_metrics['trades'])
        
        if not trades:
            return {
                'timeframe': timeframe,
                'status': 'no_data',
                'trade_count': 0
            }
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(trades, window)
        
        return {
            'timeframe': timeframe,
            'status': 'analyzed',
            'window': window,
            'trade_count': len(trades),
            'performance_metrics': performance_metrics,
            'last_updated': timeframe_metrics['last_updated']
        }
    
    def _analyze_all_timeframes(self, window: str) -> Dict[str, Any]:
        """Analyze performance for all timeframes"""
        
        timeframe_analyses = {}
        overall_metrics = {
            'total_trades': 0,
            'total_pnl': 0.0,
            'timeframe_count': 0,
            'best_timeframe': None,
            'worst_timeframe': None
        }
        
        timeframe_performances = []
        
        for timeframe in self.timeframes:
            timeframe_analysis = self._analyze_single_timeframe(timeframe, window)
            timeframe_analyses[timeframe] = timeframe_analysis
            
            if timeframe_analysis['status'] == 'analyzed':
                overall_metrics['total_trades'] += timeframe_analysis['trade_count']
                overall_metrics['total_pnl'] += timeframe_analysis['performance_metrics']['total_pnl']
                overall_metrics['timeframe_count'] += 1
                
                timeframe_performances.append({
                    'timeframe': timeframe,
                    'total_pnl': timeframe_analysis['performance_metrics']['total_pnl'],
                    'sharpe_ratio': timeframe_analysis['performance_metrics']['sharpe_ratio']
                })
        
        # Find best and worst timeframes
        if timeframe_performances:
            best_timeframe = max(timeframe_performances, key=lambda x: x['total_pnl'])
            worst_timeframe = min(timeframe_performances, key=lambda x: x['total_pnl'])
            
            overall_metrics['best_timeframe'] = best_timeframe['timeframe']
            overall_metrics['worst_timeframe'] = worst_timeframe['timeframe']
        
        return {
            'analysis_timestamp': datetime.now(),
            'window': window,
            'timeframe_analyses': timeframe_analyses,
            'overall_metrics': overall_metrics,
            'timeframe_rankings': sorted(timeframe_performances, key=lambda x: x['total_pnl'], reverse=True)
        }
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='combined performance analysis')
    async def analyze_combined_performance(
        self,
        window: str = 'medium'
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze performance across regime-timeframe combinations.
        
        Args:
            window: Analysis window ('short', 'medium', 'long')
            
        Returns:
            Dict: Combined performance analysis results
        """
        try:
            if window not in self.rolling_windows:
                self.logger.error(f"Invalid window: {window}")
                return None
            
            combined_analyses = {}
            overall_metrics = {
                'total_combinations': 0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'best_combination': None,
                'worst_combination': None
            }
            
            combination_performances = []
            
            for regime in self.regime_names:
                for timeframe in self.timeframes:
                    key = f"{regime}_{timeframe}"
                    combined_metrics = self.combined_metrics[key]
                    trades = list(combined_metrics['trades'])
                    
                    if trades:
                        performance_metrics = self._calculate_performance_metrics(trades, window)
                        
                        combined_analyses[key] = {
                            'regime': regime,
                            'timeframe': timeframe,
                            'status': 'analyzed',
                            'window': window,
                            'trade_count': len(trades),
                            'performance_metrics': performance_metrics,
                            'last_updated': combined_metrics['last_updated']
                        }
                        
                        overall_metrics['total_combinations'] += 1
                        overall_metrics['total_trades'] += len(trades)
                        overall_metrics['total_pnl'] += performance_metrics['total_pnl']
                        
                        combination_performances.append({
                            'combination': key,
                            'regime': regime,
                            'timeframe': timeframe,
                            'total_pnl': performance_metrics['total_pnl'],
                            'sharpe_ratio': performance_metrics['sharpe_ratio'],
                            'win_rate': performance_metrics['win_rate']
                        })
                    else:
                        combined_analyses[key] = {
                            'regime': regime,
                            'timeframe': timeframe,
                            'status': 'no_data',
                            'trade_count': 0
                        }
            
            # Find best and worst combinations
            if combination_performances:
                best_combination = max(combination_performances, key=lambda x: x['total_pnl'])
                worst_combination = min(combination_performances, key=lambda x: x['total_pnl'])
                
                overall_metrics['best_combination'] = best_combination['combination']
                overall_metrics['worst_combination'] = worst_combination['combination']
            
            return {
                'analysis_timestamp': datetime.now(),
                'window': window,
                'combined_analyses': combined_analyses,
                'overall_metrics': overall_metrics,
                'combination_rankings': sorted(combination_performances, key=lambda x: x['total_pnl'], reverse=True)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing combined performance: {e}")
            return None
    
    def _calculate_performance_metrics(self, trades: List[float], window: str) -> Dict[str, Any]:
        """Calculate performance metrics for a list of trades"""
        
        if not trades:
            return {
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'std_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
        
        trades_array = np.array(trades)
        
        # Basic metrics
        total_pnl = np.sum(trades_array)
        avg_pnl = np.mean(trades_array)
        std_pnl = np.std(trades_array)
        win_rate = np.sum(trades_array > 0) / len(trades_array)
        
        # Sharpe ratio (annualized for high-frequency trading)
        if std_pnl > 0:
            # Annualize based on timeframe frequency (5m = 12 intervals/hour, 15m = 4 intervals/hour, 30m = 2 intervals/hour, 1h = 1 interval/hour)
            # Use 15m as baseline (4 intervals/hour)
            sharpe_ratio = avg_pnl / std_pnl * np.sqrt(252 * 24 * 4)  # 15m intervals baseline
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum(trades_array)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(trades_array[trades_array > 0])
        gross_loss = abs(np.sum(trades_array[trades_array < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'std_pnl': std_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'total_trades': len(trades_array),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def get_attribution_summary(self) -> Dict[str, Any]:
        """Get summary of attribution system"""
        
        total_trades = len(self.trade_attributions)
        
        # Calculate summary statistics
        regime_summary = {}
        for regime, metrics in self.regime_metrics.items():
            regime_summary[regime] = {
                'trade_count': metrics['trade_count'],
                'total_pnl': metrics['total_pnl'],
                'last_updated': metrics['last_updated']
            }
        
        timeframe_summary = {}
        for timeframe, metrics in self.timeframe_metrics.items():
            timeframe_summary[timeframe] = {
                'trade_count': metrics['trade_count'],
                'total_pnl': metrics['total_pnl'],
                'last_updated': metrics['last_updated']
            }
        
        return {
            'system_status': 'active',
            'total_trades_tracked': total_trades,
            'regime_count': len(self.regime_names),
            'timeframe_count': len(self.timeframes),
            'leverage_multiplier': self.leverage_multiplier,
            'rolling_windows': self.rolling_windows,
            'regime_summary': regime_summary,
            'timeframe_summary': timeframe_summary,
            'last_updated': datetime.now()
        }