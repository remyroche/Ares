"""
TAS Regime Unified Backtesting Integration

This module provides integration between TAS regime detection and the
unified backtesting framework, replacing the legacy TAS backtesting components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import unified backtesting framework
try:
    from src.utils.common_backtesting import (
        UnifiedBacktestingOrchestrator,
        OrchestratorConfig,
        BacktestingConfig,
        BacktestingMode,
        UnifiedBacktestingResult
    )
    UNIFIED_BACKTESTING_AVAILABLE = True
except ImportError:
    UNIFIED_BACKTESTING_AVAILABLE = False

# Import TAS regime components
from ..core.tas_regime_detector import TASRegimeDetector, TASRegimeConfig

logger = logging.getLogger(__name__)


@dataclass
class TASBacktestingConfig:
    """Configuration for TAS regime backtesting integration."""
    
    # TAS-specific parameters
    tas_regime_config: TASRegimeConfig = None
    enable_tree_optimization: bool = True
    enable_hardware_optimization: bool = True
    
    # Backtesting parameters
    backtesting_config: BacktestingConfig = None
    
    # Integration parameters
    regime_weight: float = 0.5  # Weight for regime-based decisions
    confidence_threshold: float = 0.7
    
    # Output parameters
    save_tas_results: bool = True
    enable_tas_visualization: bool = True


class TASUnifiedBacktestingIntegration:
    """
    Integration between TAS regime detection and unified backtesting framework.
    
    This class replaces the legacy TAS backtesting components with the unified
    framework while maintaining TAS-specific functionality.
    """
    
    def __init__(self, config: TASBacktestingConfig):
        """Initialize TAS unified backtesting integration."""
        if not UNIFIED_BACKTESTING_AVAILABLE:
            raise ImportError("Unified backtesting framework not available")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize TAS regime detector
        if self.config.tas_regime_config:
            self.tas_detector = TASRegimeDetector(self.config.tas_regime_config)
        else:
            self.tas_detector = TASRegimeDetector()
        
        # Initialize unified backtesting orchestrator
        self.orchestrator_config = self._create_orchestrator_config()
        self.orchestrator = UnifiedBacktestingOrchestrator(self.orchestrator_config)
        
        self.logger.info("TAS unified backtesting integration initialized")
    
    def _create_orchestrator_config(self) -> OrchestratorConfig:
        """Create orchestrator configuration for TAS backtesting."""
        # Use provided backtesting config or create default
        if self.config.backtesting_config:
            backtesting_config = self.config.backtesting_config
        else:
            backtesting_config = BacktestingConfig(
                mode=BacktestingMode.HISTORICAL,
                enable_regime_detection=True,
                regime_detection_method="tas"
            )
        
        return OrchestratorConfig(
            backtesting_config=backtesting_config,
            enable_monte_carlo=True,
            enable_walk_forward=True,
            enable_performance_attribution=True,
            enable_risk_analysis=True,
            enable_regime_analysis=True,
            save_all_results=self.config.save_tas_results,
            results_directory="tas_backtesting_results",
            generate_reports=True
        )
    
    def run_tas_backtest(
        self,
        model: Any,
        data: Optional[pd.DataFrame] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> UnifiedBacktestingResult:
        """
        Run comprehensive TAS backtesting using unified framework.
        
        Args:
            model: TAS model or strategy
            data: Market data (optional)
            benchmark_data: Benchmark data for comparison (optional)
            
        Returns:
            UnifiedBacktestingResult with TAS-specific analysis
        """
        self.logger.info("Starting TAS unified backtesting")
        
        try:
            # Prepare TAS model wrapper
            tas_model = self._wrap_tas_model(model)
            
            # Run comprehensive analysis
            result = self.orchestrator.run_comprehensive_analysis(
                model=tas_model,
                data=data,
                regime_detector=self.tas_detector,
                benchmark_data=benchmark_data
            )
            
            # Add TAS-specific analysis
            result = self._add_tas_analysis(result, model, data)
            
            self.logger.info("TAS unified backtesting completed")
            return result
            
        except Exception as e:
            self.logger.error(f"TAS backtesting failed: {e}")
            raise
    
    def _wrap_tas_model(self, model: Any) -> Any:
        """Wrap TAS model for unified backtesting."""
        class TASModelWrapper:
            def __init__(self, tas_model, tas_integration):
                self.tas_model = tas_model
                self.tas_integration = tas_integration
                self.logger = logging.getLogger(__name__)
            
            def predict(self, X):
                """Make predictions using TAS model."""
                try:
                    if hasattr(self.tas_model, 'predict'):
                        return self.tas_model.predict(X)
                    else:
                        # Fallback to simple strategy
                        return self.tas_integration._simple_tas_strategy(X)
                except Exception as e:
                    self.logger.warning(f"TAS prediction failed: {e}")
                    return np.zeros(len(X))
            
            def fit(self, X, y):
                """Train TAS model."""
                try:
                    if hasattr(self.tas_model, 'fit'):
                        return self.tas_model.fit(X, y)
                    else:
                        return self
                except Exception as e:
                    self.logger.warning(f"TAS training failed: {e}")
                    return self
        
        return TASModelWrapper(model, self)
    
    def _simple_tas_strategy(self, X: np.ndarray) -> np.ndarray:
        """Simple TAS strategy as fallback."""
        # This is a simplified implementation
        # In practice, this would use actual TAS logic
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Simple momentum strategy
        if X.shape[1] >= 2:
            momentum = X[:, -1] - X[:, -2]
            signals = np.where(momentum > 0, 1, -1)
        else:
            signals = np.zeros(X.shape[0])
        
        return signals
    
    def _add_tas_analysis(
        self,
        result: UnifiedBacktestingResult,
        model: Any,
        data: Optional[pd.DataFrame]
    ) -> UnifiedBacktestingResult:
        """Add TAS-specific analysis to results."""
        try:
            # Add TAS-specific metrics
            if not hasattr(result, 'tas_metrics'):
                result.tas_metrics = {}
            
            # Tree-specific metrics
            if hasattr(model, 'get_tree_metrics'):
                result.tas_metrics['tree_metrics'] = model.get_tree_metrics()
            
            # Hardware optimization metrics
            if self.config.enable_hardware_optimization:
                result.tas_metrics['hardware_metrics'] = self._get_hardware_metrics()
            
            # Regime-specific performance
            if hasattr(self.tas_detector, 'get_regime_performance'):
                result.tas_metrics['regime_performance'] = self.tas_detector.get_regime_performance()
            
            # Tree optimization metrics
            if self.config.enable_tree_optimization:
                result.tas_metrics['tree_optimization'] = self._get_tree_optimization_metrics(model)
            
        except Exception as e:
            self.logger.warning(f"Failed to add TAS analysis: {e}")
        
        return result
    
    def _get_hardware_metrics(self) -> Dict[str, Any]:
        """Get hardware optimization metrics."""
        try:
            # This would integrate with actual hardware optimization
            return {
                'memory_usage': 0,
                'cpu_utilization': 0,
                'gpu_utilization': 0,
                'optimization_level': 'standard'
            }
        except Exception as e:
            self.logger.warning(f"Failed to get hardware metrics: {e}")
            return {}
    
    def _get_tree_optimization_metrics(self, model: Any) -> Dict[str, Any]:
        """Get tree optimization metrics."""
        try:
            # This would integrate with actual tree optimization
            return {
                'tree_depth': 0,
                'tree_nodes': 0,
                'optimization_iterations': 0,
                'convergence_score': 0.0
            }
        except Exception as e:
            self.logger.warning(f"Failed to get tree optimization metrics: {e}")
            return {}
    
    def run_tas_regime_analysis(
        self,
        data: pd.DataFrame,
        model: Any
    ) -> Dict[str, Any]:
        """Run TAS-specific regime analysis."""
        try:
            # Detect regimes using TAS
            regime_predictions = self.tas_detector.detect_regimes(data)
            
            # Analyze performance by regime
            regime_performance = {}
            for regime in np.unique(regime_predictions):
                regime_mask = regime_predictions == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) > 10:  # Minimum data points
                    regime_performance[regime] = {
                        'n_observations': len(regime_data),
                        'avg_return': regime_data.get('returns', pd.Series([0])).mean(),
                        'volatility': regime_data.get('returns', pd.Series([0])).std(),
                        'regime_stability': self._calculate_regime_stability(regime_predictions, regime)
                    }
            
            return {
                'regime_predictions': regime_predictions,
                'regime_performance': regime_performance,
                'regime_transitions': self._analyze_regime_transitions(regime_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"TAS regime analysis failed: {e}")
            return {}
    
    def _calculate_regime_stability(self, predictions: np.ndarray, regime: int) -> float:
        """Calculate stability of a specific regime."""
        regime_sequence = (predictions == regime).astype(int)
        
        # Calculate average duration of regime periods
        changes = np.diff(regime_sequence)
        regime_starts = np.where(changes == 1)[0]
        regime_ends = np.where(changes == -1)[0]
        
        if len(regime_starts) == 0:
            return 0.0
        
        durations = []
        for start in regime_starts:
            end = regime_ends[regime_ends > start]
            if len(end) > 0:
                durations.append(end[0] - start)
            else:
                durations.append(len(regime_sequence) - start)
        
        return np.mean(durations) if durations else 0.0
    
    def _analyze_regime_transitions(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze transitions between regimes."""
        transitions = {}
        unique_regimes = np.unique(predictions)
        
        for i in range(len(unique_regimes)):
            for j in range(len(unique_regimes)):
                if i != j:
                    transition_key = f"{unique_regimes[i]}_to_{unique_regimes[j]}"
                    transitions[transition_key] = 0
        
        # Count transitions
        for k in range(1, len(predictions)):
            if predictions[k] != predictions[k-1]:
                transition_key = f"{predictions[k-1]}_to_{predictions[k]}"
                if transition_key in transitions:
                    transitions[transition_key] += 1
        
        return transitions
    
    def generate_tas_report(self, result: UnifiedBacktestingResult) -> str:
        """Generate TAS-specific report."""
        report = []
        report.append("=" * 80)
        report.append("TAS REGIME UNIFIED BACKTESTING REPORT")
        report.append("=" * 80)
        
        # Overall performance
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"Total Return: {result.backtesting_result.total_return:.2%}")
        report.append(f"Sharpe Ratio: {result.backtesting_result.sharpe_ratio:.3f}")
        report.append(f"Max Drawdown: {result.backtesting_result.max_drawdown:.2%}")
        report.append(f"Overall Score: {result.overall_score:.3f}")
        
        # TAS-specific metrics
        if hasattr(result, 'tas_metrics') and result.tas_metrics:
            report.append(f"\nTAS-SPECIFIC METRICS:")
            
            if 'tree_metrics' in result.tas_metrics:
                tree_metrics = result.tas_metrics['tree_metrics']
                report.append(f"Tree Metrics: {tree_metrics}")
            
            if 'hardware_metrics' in result.tas_metrics:
                hw_metrics = result.tas_metrics['hardware_metrics']
                report.append(f"Hardware Optimization: {hw_metrics}")
            
            if 'regime_performance' in result.tas_metrics:
                regime_perf = result.tas_metrics['regime_performance']
                report.append(f"Regime Performance: {regime_perf}")
        
        # Risk analysis
        if result.risk_metrics:
            report.append(f"\nRISK ANALYSIS:")
            report.append(f"VaR (95%): {result.risk_metrics.var_95:.2%}")
            report.append(f"CVaR (95%): {result.risk_metrics.cvar_95:.2%}")
            report.append(f"Realized Volatility: {result.risk_metrics.realized_volatility:.2%}")
        
        # Walk-forward analysis
        if result.walk_forward_result:
            report.append(f"\nWALK-FORWARD ANALYSIS:")
            report.append(f"Performance Stability: {result.walk_forward_result.performance_stability:.3f}")
            report.append(f"Parameter Stability: {result.walk_forward_result.parameter_stability:.3f}")
        
        report.append(f"\nExecution Time: {result.execution_time:.2f} seconds")
        report.append("=" * 80)
        
        return "\n".join(report)


# Convenience functions for backward compatibility
def run_tas_backtest_with_unified_framework(
    model: Any,
    data: pd.DataFrame,
    config: Optional[TASBacktestingConfig] = None
) -> UnifiedBacktestingResult:
    """Run TAS backtesting using unified framework (backward compatibility)."""
    if config is None:
        config = TASBacktestingConfig()
    
    integration = TASUnifiedBacktestingIntegration(config)
    return integration.run_tas_backtest(model, data)


def create_tas_backtesting_config() -> TASBacktestingConfig:
    """Create default TAS backtesting configuration."""
    return TASBacktestingConfig(
        tas_regime_config=TASRegimeConfig(),
        enable_tree_optimization=True,
        enable_hardware_optimization=True,
        regime_weight=0.5,
        confidence_threshold=0.7
    )