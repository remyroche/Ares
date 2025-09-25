"""
Walk-Forward Analysis for TAS (ML Common)

This module provides a simplified interface to the consolidated walk-forward analyzer
for tree architecture search backtesting in the ML common utilities.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime

# Import the consolidated walk-forward analyzer
from src.utils.nas_tas.walk_forward_analyzer import (
    WalkForwardAnalyzer as ConsolidatedWalkForwardAnalyzer,
    WalkForwardConfig as ConsolidatedWalkForwardConfig,
    WalkForwardResult as ConsolidatedWalkForwardResult,
    WalkForwardMode
)

logger = logging.getLogger(__name__)


# Legacy configuration class for backward compatibility
class WalkForwardConfig:
    """Legacy configuration for walk-forward analysis - maps to consolidated config."""
    
    def __init__(self, 
                 training_window: int = 252,
                 testing_window: int = 63,
                 step_size: int = 21,
                 mode: str = "rolling",
                 min_sharpe_ratio: float = 0.5,
                 max_drawdown_threshold: float = 0.15,
                 min_win_rate: float = 0.4,
                 save_individual_results: bool = True,
                 save_summary: bool = True,
                 results_directory: str = "walk_forward_results"):
        
        # Map legacy mode to consolidated mode
        mode_mapping = {
            "rolling": WalkForwardMode.ROLLING_WINDOW,
            "expanding": WalkForwardMode.EXPANDING_WINDOW,
            "fixed": WalkForwardMode.FIXED_WINDOW
        }
        
        # Convert to consolidated config
        self.consolidated_config = ConsolidatedWalkForwardConfig(
            mode=mode_mapping.get(mode, WalkForwardMode.EXPANDING_WINDOW),
            initial_training_size=training_window,
            validation_size=testing_window,
            step_size=step_size,
            performance_threshold=min_sharpe_ratio,
            degradation_threshold=max_drawdown_threshold,
            save_results=save_summary,
            results_path=results_directory
        )
        
        # Store legacy parameters for compatibility
        self.training_window = training_window
        self.testing_window = testing_window
        self.step_size = step_size
        self.mode = mode
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_win_rate = min_win_rate
        self.save_individual_results = save_individual_results
        self.save_summary = save_summary
        self.results_directory = results_directory


# Legacy result class for backward compatibility
class WalkForwardResult:
    """Legacy result class - wraps consolidated result."""
    
    def __init__(self, consolidated_result: ConsolidatedWalkForwardResult, legacy_config: WalkForwardConfig):
        self.consolidated_result = consolidated_result
        self.legacy_config = legacy_config
        
        # Map consolidated results to legacy interface
        if consolidated_result.success:
            self.n_periods = consolidated_result.total_folds
            self.successful_periods = consolidated_result.successful_folds
            self.failed_periods = consolidated_result.total_folds - consolidated_result.successful_folds
            self.success_rate = consolidated_result.successful_folds / consolidated_result.total_folds if consolidated_result.total_folds > 0 else 0.0
            
            # Performance metrics from overall_performance
            overall_perf = consolidated_result.overall_performance
            self.average_return = overall_perf.get('accuracy', {}).get('mean', 0.0)
            self.average_sharpe = overall_perf.get('sharpe_ratio', {}).get('mean', 0.0)
            self.average_drawdown = overall_perf.get('max_drawdown', {}).get('mean', 0.0)
            self.total_return = self.average_return
            self.cumulative_return = self.total_return
            
            # Risk metrics
            self.volatility = overall_perf.get('sharpe_ratio', {}).get('std', 0.0)
            self.max_drawdown = overall_perf.get('max_drawdown', {}).get('min', 0.0)
            self.var_95 = 0.0  # Not directly available in consolidated result
            self.cvar_95 = 0.0  # Not directly available in consolidated result
            
            # Period results
            self.period_results = consolidated_result.fold_performance
            self.period_returns = [f.get('performance_metrics', {}).get('accuracy', 0.0) for f in consolidated_result.fold_performance if f.get('success', False)]
            self.period_sharpe = [f.get('performance_metrics', {}).get('sharpe_ratio', 0.0) for f in consolidated_result.fold_performance if f.get('success', False)]
            self.period_drawdown = [f.get('performance_metrics', {}).get('max_drawdown', 0.0) for f in consolidated_result.fold_performance if f.get('success', False)]
            
            # Regime analysis
            self.regime_performance = {str(k): v.get('mean_accuracy', 0.0) for k, v in consolidated_result.regime_performance.items()}
            self.regime_stability = {str(k): v for k, v in consolidated_result.regime_stability.items()}
            
            # Time series (create dummy series for compatibility)
            import pandas as pd
            import numpy as np
            n_points = len(self.period_returns) if self.period_returns else 1
            self.equity_curve = pd.Series([1.0 + sum(self.period_returns[:i+1]) for i in range(n_points)])
            self.returns_series = pd.Series(self.period_returns if self.period_returns else [0.0])
            self.drawdown_series = pd.Series([min(0, r) for r in self.period_returns] if self.period_returns else [0.0])
            
            # Metadata
            self.analysis_period = (datetime.now(), datetime.now())  # Placeholder
            self.execution_time = consolidated_result.execution_time
            self.config = legacy_config
        else:
            # Handle failure case
            self.n_periods = 0
            self.successful_periods = 0
            self.failed_periods = 0
            self.success_rate = 0.0
            self.average_return = 0.0
            self.average_sharpe = 0.0
            self.average_drawdown = 0.0
            self.total_return = 0.0
            self.cumulative_return = 0.0
            self.volatility = 0.0
            self.max_drawdown = 0.0
            self.var_95 = 0.0
            self.cvar_95 = 0.0
            self.period_results = []
            self.period_returns = []
            self.period_sharpe = []
            self.period_drawdown = []
            self.regime_performance = {}
            self.regime_stability = {}
            
            import pandas as pd
            self.equity_curve = pd.Series([1.0])
            self.returns_series = pd.Series([0.0])
            self.drawdown_series = pd.Series([0.0])
            
            self.analysis_period = (datetime.now(), datetime.now())
            self.execution_time = consolidated_result.execution_time
            self.config = legacy_config


class WalkForwardAnalyzer:
    """
    Legacy walk-forward analyzer for TAS (ML Common).
    
    This class provides backward compatibility by wrapping the consolidated
    walk-forward analyzer from src.utils.nas_tas.walk_forward_analyzer
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize walk-forward analyzer.
        
        Args:
            config: Walk-forward configuration
        """
        self.legacy_config = config
        self.consolidated_analyzer = ConsolidatedWalkForwardAnalyzer(config.consolidated_config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("✅ Legacy Walk-Forward Analyzer initialized (ML Common - using consolidated analyzer)")
        self.logger.info(f"📅 Training window: {config.training_window} days")
        self.logger.info(f"📅 Testing window: {config.testing_window} days")
        self.logger.info(f"📅 Step size: {config.step_size} days")
    
    def run_analysis(self, 
                    market_data: pd.DataFrame,
                    strategy_function: Optional[Callable] = None,
                    benchmark_data: Optional[pd.DataFrame] = None) -> WalkForwardResult:
        """
        Run comprehensive walk-forward analysis.
        
        Args:
            market_data: Historical market data (OHLCV)
            strategy_function: Optional custom strategy function
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Walk-forward analysis result
        """
        self.logger.info("🚀 Starting legacy walk-forward analysis (ML Common - delegating to consolidated analyzer)")
        
        try:
            # Run consolidated analysis
            consolidated_result = self.consolidated_analyzer.run_walk_forward_analysis(
                market_data=market_data,
                target_variable='close',  # Default target
                feature_columns=None  # Will be auto-determined
            )
            
            # Wrap result in legacy interface
            legacy_result = WalkForwardResult(consolidated_result, self.legacy_config)
            
            self.logger.info(f"✅ Legacy walk-forward analysis completed")
            self.logger.info(f"📊 Success rate: {legacy_result.success_rate:.2%}")
            self.logger.info(f"📈 Average Sharpe: {legacy_result.average_sharpe:.3f}")
            self.logger.info(f"📉 Max drawdown: {legacy_result.max_drawdown:.2%}")
            
            return legacy_result
            
        except Exception as e:
            self.logger.error(f"❌ Legacy walk-forward analysis failed: {e}")
            raise
    
    def get_results(self) -> Optional[WalkForwardResult]:
        """Get walk-forward analysis results."""
        if hasattr(self.consolidated_analyzer, 'results') and self.consolidated_analyzer.results:
            return WalkForwardResult(self.consolidated_analyzer.results, self.legacy_config)
        return None
    
    def export_results(self, filepath: str):
        """Export results to file."""
        self.logger.warning("⚠️ Export functionality delegated to consolidated analyzer")
        # The consolidated analyzer handles saving internally
        pass