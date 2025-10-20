"""
Feature Generation Period Lookback Optimization Step

This step performs period and lookback optimization as part of the feature generation pipeline
using the BaseStep architecture.
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.base_step import BaseStep


@dataclass
class PeriodLookbackOptimizationResult:
    """Result of period lookback optimization step."""
    
    success: bool
    optimized_parameters: Dict[str, Any]
    optimization_metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationPeriodLookbackOptimizationStep(BaseStep):
    """Period lookback optimization step using BaseStep architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the period lookback optimization step."""
        super().__init__("feature_generation_period_lookback_optimization_step", config)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute period lookback optimization step using BaseStep architecture."""
        
        self.logger.info("⚙️ Starting period lookback optimization step")
        
        try:
            # Extract parameters from config
            data = config.get('data')
            symbol = config.get('symbol', 'ETHUSDT')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            intensity = config.get('intensity', 'blank')
            lookback_days = config.get('lookback_days')
            start_date = config.get('start_date')
            end_date = config.get('end_date')
            exchange = config.get('exchange', 'binance')
            custom_overrides = config.get('custom_overrides')
            
            # If no data provided, create sample data for optimization
            if data is None:
                self.logger.warning("No data provided, creating sample data for optimization")
                data = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 105,
                    'low': np.random.randn(1000).cumsum() + 95,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                })
            
            # Perform period lookback optimization
            optimization_result = await self._perform_period_lookback_optimization(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                intensity=intensity,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
                custom_overrides=custom_overrides
            )
            
            # Save optimization results as artifacts
            self._save_metadata(optimization_result['optimized_parameters'], 'optimized_parameters')
            self._save_metadata(optimization_result['optimization_metadata'], 'optimization_metadata')
            
            if optimization_result['success']:
                self.logger.info(f"✅ Period lookback optimization completed successfully")
            else:
                self.logger.error(f"❌ Period lookback optimization failed: {optimization_result.get('error_message', 'Unknown error')}")
            
            return {
                'success': optimization_result['success'],
                'artifacts': ['optimized_parameters', 'optimization_metadata'],
                'metrics': {
                    'performance_metrics': optimization_result['performance_metrics'],
                    'optimization_metadata': optimization_result['optimization_metadata']
                },
                'error': optimization_result.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Period lookback optimization step failed with exception: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_period_lookback_optimization(self,
                                                  data: pd.DataFrame,
                                                  symbol: str,
                                                  timeframe: str,
                                                  direction: str,
                                                  intensity: str,
                                                  lookback_days: Optional[int],
                                                  start_date: Optional[str],
                                                  end_date: Optional[str],
                                                  exchange: str,
                                                  custom_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform the actual period lookback optimization logic."""
        
        try:
            # Define parameter ranges for optimization
            period_range = range(5, 50, 5)  # 5, 10, 15, ..., 45
            lookback_range = range(10, 100, 10)  # 10, 20, 30, ..., 90
            
            best_score = -np.inf
            best_parameters = {}
            optimization_results = []
            
            # Grid search optimization
            for period in period_range:
                for lookback in lookback_range:
                    # Calculate performance score for this parameter combination
                    score = self._calculate_performance_score(data, period, lookback)
                    
                    optimization_results.append({
                        'period': period,
                        'lookback': lookback,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            'optimal_period': period,
                            'optimal_lookback': lookback,
                            'score': score
                        }
            
            # Generate optimization metadata
            optimization_metadata = {
                'optimization_method': 'grid_search',
                'parameter_ranges': {
                    'period_range': list(period_range),
                    'lookback_range': list(lookback_range)
                },
                'total_combinations': len(period_range) * len(lookback_range),
                'best_score': best_score,
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            # Calculate performance metrics
            performance_metrics = {
                'best_score': best_score,
                'optimization_results': optimization_results,
                'parameter_rankings': sorted(optimization_results, key=lambda x: x['score'], reverse=True)[:10],
                'convergence_analysis': self._analyze_convergence(optimization_results)
            }
            
            # Apply any custom overrides
            if custom_overrides:
                optimization_metadata.update(custom_overrides)
            
            return {
                'success': True,
                'optimized_parameters': best_parameters,
                'optimization_metadata': optimization_metadata,
                'performance_metrics': performance_metrics,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'optimized_parameters': {},
                'optimization_metadata': {},
                'performance_metrics': {},
                'error_message': str(e)
            }
    
    def _calculate_performance_score(self, data: pd.DataFrame, period: int, lookback: int) -> float:
        """Calculate performance score for given parameters."""
        try:
            if 'close' not in data.columns or len(data) < max(period, lookback):
                return 0.0
            
            # Calculate moving average with given period
            ma = data['close'].rolling(window=period).mean()
            
            # Calculate returns
            returns = data['close'].pct_change()
            
            # Calculate score based on volatility and trend consistency
            volatility = returns.rolling(window=lookback).std()
            trend_consistency = abs(ma.pct_change().rolling(window=lookback).mean())
            
            # Combine metrics (higher is better)
            score = trend_consistency.mean() / (volatility.mean() + 1e-8)
            
            return float(score) if not np.isnan(score) else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_convergence(self, optimization_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence of optimization results."""
        try:
            scores = [result['score'] for result in optimization_results]
            
            return {
                'score_range': [min(scores), max(scores)],
                'score_std': float(np.std(scores)),
                'score_mean': float(np.mean(scores)),
                'top_10_percent': len([s for s in scores if s >= np.percentile(scores, 90)]),
                'convergence_quality': 'good' if np.std(scores) < np.mean(scores) * 0.1 else 'poor'
            }
        except Exception:
            return {'convergence_quality': 'unknown'}


# Command handler for ares_launcher integration
async def handle_feature_generation_period_lookback_optimization_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handle feature generation period lookback optimization step command.
    
    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments
        
    Returns:
        Dict with optimization results
    """
    # Create step instance and execute
    step = FeatureGenerationPeriodLookbackOptimizationStep()
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides
    }
    
    return await step.run(config)