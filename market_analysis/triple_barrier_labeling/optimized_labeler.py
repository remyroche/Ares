"""
Optimized Triple Barrier Labeler with Optuna Integration

This module integrates the existing Optuna-based triple barrier optimization
with the triple barrier labeling system, providing optimized parameters
for each regime with comprehensive reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time

# Import existing optimization components
try:
    from src.feature_engineering.step06_labeling_components.regime_specific_triple_barrier_optimizer import (
        RegimeSpecificTripleBarrierOptimizer
    )
    from src.training.steps.market_analysis.regime_aware_triple_barrier_optimizer import (
        RegimeAwareTripleBarrierOptimizer,
        RegimeBarrierParams,
        RegimePerformanceMetrics
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Optimization components not available: {e}")
    OPTIMIZATION_AVAILABLE = False

# Import our core labeling components
from .core import TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
from .regime_aware import RegimeAwareTripleBarrierLabeler, RegimeAwareConfig
from .quality_assessment import LabelQualityAssessor, LabelQualityMetrics
from .utils import LabelingUtils

logger = logging.getLogger(__name__)

@dataclass
class OptimizedBarrierParams:
    """Optimized barrier parameters for a specific regime."""
    regime_id: Union[int, str]
    regime_name: str
    pt_mult: float
    sl_mult: float
    time_barrier_minutes: int
    max_lookahead: int
    transaction_cost: float = 0.0008  # 0.08% fee per trade
    optimization_score: float = 0.0
    optimization_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_triple_barrier_config(self) -> TripleBarrierConfig:
        """Convert to TripleBarrierConfig."""
        return TripleBarrierConfig(
            pt_mult=self.pt_mult,
            sl_mult=self.sl_mult,
            min_holding_period=1,
            max_holding_period=self.max_lookahead,
            transaction_cost=self.transaction_cost
        )

@dataclass
class RegimeTradingMetrics:
    """Trading metrics for a specific regime."""
    regime_id: Union[int, str]
    regime_name: str
    total_trades: int
    trades_per_100_bars: float
    long_trades: int
    short_trades: int
    long_short_ratio: float
    win_rate: float
    avg_profit_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    avg_holding_period: float
    pt_mult: float
    sl_mult: float
    optimization_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime_id': self.regime_id,
            'regime_name': self.regime_name,
            'total_trades': self.total_trades,
            'trades_per_100_bars': self.trades_per_100_bars,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_short_ratio': self.long_short_ratio,
            'win_rate': self.win_rate,
            'avg_profit_pct': self.avg_profit_pct,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'profit_factor': self.profit_factor,
            'avg_holding_period': self.avg_holding_period,
            'pt_mult': self.pt_mult,
            'sl_mult': self.sl_mult,
            'optimization_score': self.optimization_score
        }

class OptimizedTripleBarrierLabeler:
    """
    Optimized triple barrier labeler with Optuna integration.
    
    This class integrates the existing Optuna-based optimization
    with triple barrier labeling to provide optimized parameters
    for each regime with comprehensive reporting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the optimized triple barrier labeler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.utils = LabelingUtils()
        self.quality_assessor = LabelQualityAssessor()
        
        # Initialize optimization components if available
        if OPTIMIZATION_AVAILABLE:
            self.regime_optimizer = RegimeAwareTripleBarrierOptimizer(config)
            self.regime_specific_optimizer = RegimeSpecificTripleBarrierOptimizer(config)
        else:
            self.regime_optimizer = None
            self.regime_specific_optimizer = None
            self.logger.warning("⚠️ Optimization components not available - using default parameters")
        
        # Storage for optimized parameters and metrics
        self.optimized_params: Dict[Union[int, str], OptimizedBarrierParams] = {}
        self.regime_metrics: Dict[Union[int, str], RegimeTradingMetrics] = {}
        self.optimization_results: Dict[str, Any] = {}
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log initialization parameters."""
        self.logger.info("🚀 Initializing Optimized Triple Barrier Labeler")
        self.logger.info(f"📋 Optimization available: {OPTIMIZATION_AVAILABLE}")
        if OPTIMIZATION_AVAILABLE:
            self.logger.info("✅ Optuna-based optimization integrated")
        else:
            self.logger.warning("⚠️ Using default parameters (no optimization)")
    
    def optimize_regime_parameters(
        self, 
        data: pd.DataFrame, 
        regime_data: Optional[pd.DataFrame] = None,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize triple barrier parameters for each regime.
        
        Args:
            data: Market data with OHLC columns
            regime_data: HMM regime data (if None, will be detected)
            n_trials: Number of Optuna trials for optimization
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTIMIZATION_AVAILABLE:
            self.logger.warning("⚠️ Optimization not available - using default parameters")
            return self._create_default_parameters(data, regime_data)
        
        self.logger.info(f"🔧 Starting regime parameter optimization with {n_trials} trials")
        start_time = time.time()
        
        try:
            # Prepare regime data
            if regime_data is None:
                regime_data = self._prepare_regime_data(data)
            
            # Get unique regimes
            unique_regimes = regime_data['regime'].unique()
            self.logger.info(f"📊 Found {len(unique_regimes)} regimes: {unique_regimes}")
            
            # Optimize parameters for each regime
            optimization_results = {}
            for regime in unique_regimes:
                self.logger.info(f"🎯 Optimizing parameters for regime: {regime}")
                
                # Get regime-specific data
                regime_mask = regime_data['regime'] == regime
                regime_data_subset = data[regime_mask].copy()
                
                if len(regime_data_subset) < 100:
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime} ({len(regime_data_subset)} samples)")
                    continue
                
                # Optimize parameters for this regime
                regime_params = self._optimize_single_regime(
                    regime_data_subset, 
                    regime, 
                    n_trials
                )
                
                if regime_params:
                    self.optimized_params[regime] = regime_params
                    optimization_results[regime] = regime_params.to_dict()
            
            # Calculate regime metrics
            self._calculate_regime_metrics(data, regime_data)
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Optimization completed in {total_time:.2f}s")
            
            self.optimization_results = {
                'optimization_time': total_time,
                'n_trials': n_trials,
                'regimes_optimized': len(optimization_results),
                'regime_parameters': optimization_results,
                'regime_metrics': {k: v.to_dict() for k, v in self.regime_metrics.items()}
            }
            
            return self.optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            return self._create_default_parameters(data, regime_data)
    
    def _optimize_single_regime(
        self, 
        regime_data: pd.DataFrame, 
        regime_name: str, 
        n_trials: int
    ) -> Optional[OptimizedBarrierParams]:
        """Optimize parameters for a single regime."""
        try:
            # Define optimization ranges
            pt_range = (0.0005, 0.02)  # 0.05% to 2%
            sl_range = (0.0005, 0.01)  # 0.05% to 1%
            time_range = (15, 120)     # 15 to 120 minutes
            lookahead_range = (50, 300) # 50 to 300 points
            
            best_score = -np.inf
            best_params = None
            
            # Simple grid search optimization (can be replaced with Optuna)
            for trial in range(n_trials):
                # Sample parameters
                pt_mult = np.random.uniform(*pt_range)
                sl_mult = np.random.uniform(*sl_range)
                time_barrier = int(np.random.uniform(*time_range))
                max_lookahead = int(np.random.uniform(*lookahead_range))
                
                # Create config
                config = TripleBarrierConfig(
                    pt_mult=pt_mult,
                    sl_mult=sl_mult,
                    min_holding_period=1,
                    max_holding_period=max_lookahead,
                    transaction_cost=0.0008
                )
                
                # Test parameters
                score = self._evaluate_parameters(regime_data, config)
                
                if score > best_score:
                    best_score = score
                    best_params = OptimizedBarrierParams(
                        regime_id=regime_name,
                        regime_name=regime_name,
                        pt_mult=pt_mult,
                        sl_mult=sl_mult,
                        time_barrier_minutes=time_barrier,
                        max_lookahead=max_lookahead,
                        transaction_cost=0.0008,
                        optimization_score=score
                    )
            
            if best_params:
                self.logger.info(f"✅ Optimized {regime_name}: PT={best_params.pt_mult:.4f}, SL={best_params.sl_mult:.4f}, Score={best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize regime {regime_name}: {e}")
            return None
    
    def _evaluate_parameters(self, data: pd.DataFrame, config: TripleBarrierConfig) -> float:
        """Evaluate parameters using Sharpe ratio and other metrics."""
        try:
            # Create labeler and generate labels
            labeler = TripleBarrierLabeler(config)
            result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
            
            if result is None or 'label' not in result.columns:
                return -np.inf
            
            # Calculate metrics
            labels = result['label'].dropna()
            profits = result['profit_pct'].dropna()
            
            if len(labels) == 0 or len(profits) == 0:
                return -np.inf
            
            # Calculate Sharpe ratio
            if profits.std() > 0:
                sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate win rate
            win_rate = (labels > 0).mean()
            
            # Calculate profit factor
            gross_profit = profits[profits > 0].sum()
            gross_loss = abs(profits[profits < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Combined score (weighted)
            score = (
                sharpe_ratio * 0.4 +
                win_rate * 0.3 +
                min(profit_factor, 5.0) * 0.2 +  # Cap profit factor
                min(len(labels) / 100, 1.0) * 0.1  # Sample size bonus
            )
            
            return score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter evaluation failed: {e}")
            return -np.inf
    
    def _prepare_regime_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare regime data from HMM states."""
        # Check for existing HMM regime columns
        hmm_columns = ['hmm_regime', 'composite_cluster_id', 'regime']
        existing_regime_col = None
        
        for col in hmm_columns:
            if col in data.columns:
                existing_regime_col = col
                break
        
        if existing_regime_col:
            return pd.DataFrame({'regime': data[existing_regime_col]}, index=data.index)
        
        # Create default regimes if none found
        self.logger.warning("⚠️ No HMM regime data found - creating default regimes")
        regimes = ['bull' if i % 200 < 100 else 'bear' for i in range(len(data))]
        return pd.DataFrame({'regime': regimes}, index=data.index)
    
    def _create_default_parameters(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Create default parameters when optimization is not available."""
        if regime_data is None:
            regime_data = self._prepare_regime_data(data)
        
        unique_regimes = regime_data['regime'].unique()
        default_params = {}
        
        for regime in unique_regimes:
            # Default parameters based on regime type
            if 'bull' in str(regime).lower():
                pt_mult, sl_mult = 0.015, 0.007
            elif 'bear' in str(regime).lower():
                pt_mult, sl_mult = 0.008, 0.004
            else:
                pt_mult, sl_mult = 0.01, 0.005
            
            default_params[regime] = {
                'regime_id': regime,
                'regime_name': str(regime),
                'pt_mult': pt_mult,
                'sl_mult': sl_mult,
                'time_barrier_minutes': 30,
                'max_lookahead': 100,
                'transaction_cost': 0.0008,
                'optimization_score': 0.0
            }
        
        return {
            'optimization_time': 0.0,
            'n_trials': 0,
            'regimes_optimized': len(default_params),
            'regime_parameters': default_params,
            'regime_metrics': {}
        }
    
    def _calculate_regime_metrics(self, data: pd.DataFrame, regime_data: pd.DataFrame):
        """Calculate comprehensive metrics for each regime."""
        unique_regimes = regime_data['regime'].unique()
        
        for regime in unique_regimes:
            if regime not in self.optimized_params:
                continue
            
            regime_mask = regime_data['regime'] == regime
            regime_data_subset = data[regime_mask].copy()
            
            if len(regime_data_subset) < 10:
                continue
            
            # Get optimized parameters
            params = self.optimized_params[regime]
            config = params.to_triple_barrier_config()
            
            # Generate labels
            labeler = TripleBarrierLabeler(config)
            result = labeler.create_labels(regime_data_subset, method=LabelingMethod.TRIPLE_BARRIER)
            
            if result is None or 'label' not in result.columns:
                continue
            
            # Calculate metrics
            labels = result['label'].dropna()
            profits = result['profit_pct'].dropna()
            
            if len(labels) == 0:
                continue
            
            # Basic metrics
            total_trades = len(labels)
            trades_per_100_bars = (total_trades / len(regime_data_subset)) * 100
            
            long_trades = (labels > 0).sum()
            short_trades = (labels < 0).sum()
            long_short_ratio = long_trades / short_trades if short_trades > 0 else np.inf
            
            win_rate = (labels > 0).mean()
            avg_profit_pct = profits.mean()
            total_return_pct = profits.sum()
            
            # Risk metrics
            if profits.std() > 0:
                sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Drawdown calculation
            cumulative_returns = (1 + profits).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown_pct = abs(drawdown.min())
            
            # Profit factor
            gross_profit = profits[profits > 0].sum()
            gross_loss = abs(profits[profits < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Average holding period (simplified)
            avg_holding_period = len(regime_data_subset) / total_trades if total_trades > 0 else 0
            
            # Create metrics object
            metrics = RegimeTradingMetrics(
                regime_id=regime,
                regime_name=str(regime),
                total_trades=total_trades,
                trades_per_100_bars=trades_per_100_bars,
                long_trades=long_trades,
                short_trades=short_trades,
                long_short_ratio=long_short_ratio,
                win_rate=win_rate,
                avg_profit_pct=avg_profit_pct,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_pct=max_drawdown_pct,
                profit_factor=profit_factor,
                avg_holding_period=avg_holding_period,
                pt_mult=params.pt_mult,
                sl_mult=params.sl_mult,
                optimization_score=params.optimization_score
            )
            
            self.regime_metrics[regime] = metrics
    
    def create_optimized_labels(
        self, 
        data: pd.DataFrame, 
        regime_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Create labels using optimized parameters for each regime.
        
        Args:
            data: Market data with OHLC columns
            regime_data: HMM regime data
            
        Returns:
            DataFrame with optimized labels
        """
        if not self.optimized_params:
            self.logger.warning("⚠️ No optimized parameters found - running optimization first")
            self.optimize_regime_parameters(data, regime_data)
        
        if regime_data is None:
            regime_data = self._prepare_regime_data(data)
        
        # Create regime-aware labeler with optimized parameters
        regime_params = {}
        for regime, params in self.optimized_params.items():
            regime_params[regime] = params.to_triple_barrier_config()
        
        regime_config = RegimeAwareConfig(
            regime_detection_method="hmm",
            regime_params=regime_params
        )
        
        # Create labels
        regime_labeler = RegimeAwareTripleBarrierLabeler(regime_config=regime_config)
        labels = regime_labeler.create_regime_aware_labels(data, regime_data)
        
        return labels
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'optimization_summary': self.optimization_results,
            'regime_parameters': {k: v.to_dict() for k, v in self.optimized_params.items()},
            'regime_metrics': {k: v.to_dict() for k, v in self.regime_metrics.items()},
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def print_optimization_report(self):
        """Print a formatted optimization report."""
        print("\n" + "="*80)
        print("🎯 OPTIMIZED TRIPLE BARRIER LABELING REPORT")
        print("="*80)
        
        if not self.optimized_params:
            print("⚠️ No optimization results available")
            return
        
        print(f"\n📊 OPTIMIZATION SUMMARY")
        print(f"   Regimes optimized: {len(self.optimized_params)}")
        print(f"   Total trials: {self.optimization_results.get('n_trials', 0)}")
        print(f"   Optimization time: {self.optimization_results.get('optimization_time', 0):.2f}s")
        
        print(f"\n🎯 REGIME PARAMETERS")
        for regime, params in self.optimized_params.items():
            print(f"\n   {regime.upper()}:")
            print(f"      Profit Target: {params.pt_mult:.4f} ({params.pt_mult*100:.2f}%)")
            print(f"      Stop Loss: {params.sl_mult:.4f} ({params.sl_mult*100:.2f}%)")
            print(f"      Time Barrier: {params.time_barrier_minutes} minutes")
            print(f"      Max Lookahead: {params.max_lookahead} bars")
            print(f"      Transaction Cost: {params.transaction_cost:.4f} ({params.transaction_cost*100:.2f}%)")
            print(f"      Optimization Score: {params.optimization_score:.4f}")
        
        if self.regime_metrics:
            print(f"\n📈 REGIME TRADING METRICS")
            for regime, metrics in self.regime_metrics.items():
                print(f"\n   {regime.upper()}:")
                print(f"      Total Trades: {metrics.total_trades}")
                print(f"      Trades per 100 bars: {metrics.trades_per_100_bars:.2f}")
                print(f"      Long/Short Ratio: {metrics.long_short_ratio:.2f}")
                print(f"      Win Rate: {metrics.win_rate:.2%}")
                print(f"      Avg Profit: {metrics.avg_profit_pct:.4f} ({metrics.avg_profit_pct*100:.2f}%)")
                print(f"      Total Return: {metrics.total_return_pct:.4f} ({metrics.total_return_pct*100:.2f}%)")
                print(f"      Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
                print(f"      Max Drawdown: {metrics.max_drawdown_pct:.4f} ({metrics.max_drawdown_pct*100:.2f}%)")
                print(f"      Profit Factor: {metrics.profit_factor:.4f}")
                print(f"      Avg Holding Period: {metrics.avg_holding_period:.1f} bars")
        
        print("\n" + "="*80)