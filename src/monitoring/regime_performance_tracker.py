#!/usr/bin/env python3
"""Regime Performance Tracking System.

This module provides comprehensive tracking of trading performance segmented by market regime.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import sqlite3

from src.utils.logger import system_logger
from src.utils.common_operations import ensure_directory, safe_json_dump

logger = system_logger.getChild("RegimePerformanceTracker")


class RegimePerformanceTracker:
    """Comprehensive regime-based performance tracking system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimePerformanceTracker")
        
        # Initialize storage
        self.db_path = Path(config.get('data_dir', 'data')) / 'regime_performance.db'
        ensure_directory(self.db_path.parent)
        
        # Initialize database
        self._init_database()
        
        # Performance metrics cache
        self.metrics_cache = defaultdict(lambda: defaultdict(list))
        
        # Regime definitions
        self.regime_names = ['bull', 'bear', 'sideways', 'transition']
        
    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                regime TEXT,
                regime_confidence REAL,
                action TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                pnl_percent REAL,
                holding_period_minutes INTEGER,
                max_drawdown REAL,
                models_used TEXT,
                features_used TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                from_regime TEXT,
                to_regime TEXT,
                confidence REAL,
                detection_lag_minutes INTEGER,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date DATE,
                symbol TEXT,
                regime TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_holding_period REAL,
                PRIMARY KEY (date, symbol, regime)
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def track_trade(self, trade_data: Dict[str, Any]) -> None:
        """Track a completed trade with regime information."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, regime, regime_confidence, action,
                    entry_price, exit_price, quantity, pnl, pnl_percent,
                    holding_period_minutes, max_drawdown, models_used,
                    features_used, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['timestamp'],
                trade_data['symbol'],
                trade_data['regime'],
                trade_data.get('regime_confidence', 1.0),
                trade_data['action'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['quantity'],
                trade_data['pnl'],
                trade_data['pnl_percent'],
                trade_data.get('holding_period_minutes', 0),
                trade_data.get('max_drawdown', 0),
                json.dumps(trade_data.get('models_used', [])),
                json.dumps(trade_data.get('features_used', [])),
                json.dumps(trade_data.get('metadata', {}))
            ))
            
            conn.commit()
            
            # Update cache
            self.metrics_cache[trade_data['regime']]['pnl'].append(trade_data['pnl'])
            self.metrics_cache[trade_data['regime']]['pnl_percent'].append(trade_data['pnl_percent'])
            
        finally:
            conn.close()
    
    async def track_regime_transition(self, transition_data: Dict[str, Any]) -> None:
        """Track regime transitions."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO regime_transitions (
                    timestamp, symbol, from_regime, to_regime,
                    confidence, detection_lag_minutes, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                transition_data['timestamp'],
                transition_data['symbol'],
                transition_data['from_regime'],
                transition_data['to_regime'],
                transition_data.get('confidence', 1.0),
                transition_data.get('detection_lag_minutes', 0),
                json.dumps(transition_data.get('metadata', {}))
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def calculate_regime_metrics(self, symbol: str, 
                                     period_days: int = 30) -> Dict[str, Any]:
        """Calculate comprehensive metrics for each regime."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Query trades by regime
            query = """
                SELECT regime, pnl, pnl_percent, holding_period_minutes,
                       max_drawdown, regime_confidence
                FROM trades
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            """
            
            df = pd.read_sql_query(
                query, 
                conn,
                params=(symbol, start_date, end_date)
            )
            
            if df.empty:
                return self._empty_metrics()
            
            # Calculate metrics by regime
            metrics = {}
            
            for regime in self.regime_names:
                regime_df = df[df['regime'] == regime]
                
                if len(regime_df) == 0:
                    metrics[regime] = self._empty_regime_metrics()
                    continue
                
                # Calculate returns series for Sharpe
                returns = regime_df['pnl_percent'].values / 100
                
                metrics[regime] = {
                    'trade_count': len(regime_df),
                    'win_rate': (regime_df['pnl'] > 0).mean(),
                    'total_pnl': regime_df['pnl'].sum(),
                    'avg_pnl': regime_df['pnl'].mean(),
                    'avg_pnl_percent': regime_df['pnl_percent'].mean(),
                    'sharpe_ratio': self._calculate_sharpe(returns),
                    'sortino_ratio': self._calculate_sortino(returns),
                    'max_drawdown': regime_df['max_drawdown'].max(),
                    'avg_holding_period': regime_df['holding_period_minutes'].mean(),
                    'profit_factor': self._calculate_profit_factor(regime_df['pnl']),
                    'avg_confidence': regime_df['regime_confidence'].mean(),
                    'calmar_ratio': self._calculate_calmar(
                        regime_df['pnl'].sum(),
                        regime_df['max_drawdown'].max()
                    )
                }
            
            # Add comparative analysis
            metrics['comparison'] = self._compare_regimes(metrics)
            
            # Add transition analysis
            metrics['transitions'] = await self._analyze_transitions(symbol, start_date, end_date)
            
            return metrics
            
        finally:
            conn.close()
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
            
        # Annualized Sharpe (assuming daily returns)
        return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
            
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        
        return np.mean(returns) / downside_std * np.sqrt(365)
    
    def _calculate_profit_factor(self, pnl_series: pd.Series) -> float:
        """Calculate profit factor."""
        profits = pnl_series[pnl_series > 0].sum()
        losses = abs(pnl_series[pnl_series < 0].sum())
        
        return profits / losses if losses > 0 else float('inf')
    
    def _calculate_calmar(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        return total_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    def _compare_regimes(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare performance across regimes."""
        
        comparison = {
            'best_regime': {
                'by_sharpe': max(
                    [(r, m['sharpe_ratio']) for r, m in metrics.items() if r != 'comparison'],
                    key=lambda x: x[1]
                )[0] if metrics else None,
                'by_pnl': max(
                    [(r, m['total_pnl']) for r, m in metrics.items() if r != 'comparison'],
                    key=lambda x: x[1]
                )[0] if metrics else None,
                'by_win_rate': max(
                    [(r, m['win_rate']) for r, m in metrics.items() if r != 'comparison'],
                    key=lambda x: x[1]
                )[0] if metrics else None
            },
            'regime_ranking': sorted(
                [(r, m['sharpe_ratio']) for r, m in metrics.items() if r not in ['comparison', 'transitions']],
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        return comparison
    
    async def _analyze_transitions(self, symbol: str, start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Analyze regime transitions."""
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = """
                SELECT from_regime, to_regime, detection_lag_minutes
                FROM regime_transitions
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, start_date, end_date)
            )
            
            if df.empty:
                return {}
            
            # Calculate transition statistics
            transition_counts = df.groupby(['from_regime', 'to_regime']).size()
            avg_lag = df.groupby(['from_regime', 'to_regime'])['detection_lag_minutes'].mean()
            
            return {
                'transition_counts': transition_counts.to_dict(),
                'avg_detection_lag': avg_lag.to_dict(),
                'total_transitions': len(df)
            }
            
        finally:
            conn.close()
    
    async def generate_regime_report(self, symbol: str, 
                                   output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive regime performance report."""
        
        report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'metrics_7d': await self.calculate_regime_metrics(symbol, 7),
            'metrics_30d': await self.calculate_regime_metrics(symbol, 30),
            'metrics_90d': await self.calculate_regime_metrics(symbol, 90),
            'recommendations': []
        }
        
        # Generate recommendations based on metrics
        for period, metrics in [('7d', report['metrics_7d']), 
                               ('30d', report['metrics_30d'])]:
            for regime, regime_metrics in metrics.items():
                if regime in ['comparison', 'transitions']:
                    continue
                    
                # Low Sharpe ratio recommendation
                if regime_metrics['sharpe_ratio'] < 0.5:
                    report['recommendations'].append({
                        'regime': regime,
                        'period': period,
                        'issue': 'low_sharpe',
                        'recommendation': f"Review {regime} regime strategy - Sharpe ratio below 0.5"
                    })
                
                # Low win rate recommendation
                if regime_metrics['win_rate'] < 0.4:
                    report['recommendations'].append({
                        'regime': regime,
                        'period': period,
                        'issue': 'low_win_rate',
                        'recommendation': f"Improve {regime} regime entry signals - win rate below 40%"
                    })
        
        # Save report if path provided
        if output_path:
            ensure_directory(output_path.parent)
            safe_json_dump(report, output_path)
            self.logger.info(f"Saved regime report to {output_path}")
        
        return report
    
    async def update_daily_metrics(self) -> None:
        """Update daily aggregated metrics."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get yesterday's date
            yesterday = (datetime.now() - timedelta(days=1)).date()
            
            # Query trades from yesterday
            query = """
                SELECT symbol, regime, COUNT(*) as trade_count,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                       SUM(pnl) as total_pnl,
                       MAX(max_drawdown) as max_drawdown,
                       AVG(holding_period_minutes) as avg_holding_period
                FROM trades
                WHERE DATE(timestamp) = ?
                GROUP BY symbol, regime
            """
            
            results = cursor.execute(query, (yesterday,)).fetchall()
            
            for row in results:
                symbol, regime, trade_count, winning_trades, total_pnl, max_dd, avg_hold = row
                
                # Calculate Sharpe for the day
                pnl_query = """
                    SELECT pnl_percent FROM trades
                    WHERE DATE(timestamp) = ? AND symbol = ? AND regime = ?
                """
                pnl_data = cursor.execute(pnl_query, (yesterday, symbol, regime)).fetchall()
                returns = [p[0] / 100 for p in pnl_data]
                sharpe = self._calculate_sharpe(np.array(returns))
                
                # Insert or update daily metrics
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_metrics
                    (date, symbol, regime, total_trades, winning_trades,
                     total_pnl, sharpe_ratio, max_drawdown, avg_holding_period)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    yesterday, symbol, regime, trade_count, winning_trades,
                    total_pnl, sharpe, max_dd, avg_hold
                ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            regime: self._empty_regime_metrics() 
            for regime in self.regime_names
        }
    
    def _empty_regime_metrics(self) -> Dict[str, float]:
        """Return empty regime metrics."""
        return {
            'trade_count': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'avg_pnl_percent': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_period': 0.0,
            'profit_factor': 0.0,
            'avg_confidence': 0.0,
            'calmar_ratio': 0.0
        }


# Convenience function for tracking
async def track_trade_with_regime(trade_data: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Track a trade with regime information."""
    tracker = RegimePerformanceTracker(config)
    await tracker.track_trade(trade_data)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {'data_dir': 'data'}
        tracker = RegimePerformanceTracker(config)
        
        # Example trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': 'BTCUSDT',
            'regime': 'bull',
            'regime_confidence': 0.85,
            'action': 'long',
            'entry_price': 50000,
            'exit_price': 51000,
            'quantity': 0.1,
            'pnl': 100,
            'pnl_percent': 2.0,
            'holding_period_minutes': 120,
            'max_drawdown': 0.5,
            'models_used': ['momentum_model', 'breakout_model'],
            'features_used': ['rsi', 'macd', 'volume']
        }
        
        await tracker.track_trade(trade)
        
        # Generate report
        report = await tracker.generate_regime_report('BTCUSDT')
        print(json.dumps(report, indent=2))
    
    asyncio.run(main())