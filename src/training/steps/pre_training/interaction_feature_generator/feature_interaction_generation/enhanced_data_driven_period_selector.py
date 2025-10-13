"""
Enhanced Data-Driven Period Selector with Economic Significance Evaluation

This module extends the DataDrivenPeriodSelector to include economic significance
evaluation and backtesting, following the pattern of DataDrivenInteractionGenerator.

Key Features:
- Combines statistical analysis with economic significance evaluation
- Backtesting against financial targets (Sharpe ratio, max drawdown, win rate)
- Period ranking based on both data characteristics and economic performance
- Optimized for 15m timeframe (1-50 periods)
- VectorBT-optimized for performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

# Import existing components
from .data_driven_periods import (
    DataDrivenPeriodSelector, PeriodAnalysisResult,
    PeriodAnalyzer, PeriodValidator, PeriodSelector
)
from .economic_period_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig,
    EconomicPeriodEvaluationResult, PeriodBacktestResult
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPeriodSelectionConfig:
    """Configuration for enhanced period selection."""
    
    # Data-driven analysis configuration
    min_period: int = 1
    max_period: int = 50  # Optimized for 15m timeframe
    max_periods: int = 8
    min_data_points: int = 100
    
    # Economic evaluation configuration
    enable_economic_evaluation: bool = True
    min_economic_score: float = 0.4
    economic_weight: float = 0.6  # Weight for economic vs statistical analysis
    statistical_weight: float = 0.4
    
    # Backtesting configuration
    backtest_periods: int = 100
    min_backtest_periods: int = 50
    
    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Timeframe-specific period ranges
    timeframe_period_ranges: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        """Initialize timeframe-specific period ranges."""
        if self.timeframe_period_ranges is None:
            self.timeframe_period_ranges = {
                "5m": (1, 100),   # 5m to 8.3 hours
                "15m": (1, 50),   # 15m to 12.5 hours ✅
                "1h": (1, 24),    # 1h to 1 day
                "4h": (1, 12),    # 4h to 2 days
            }


@dataclass
class EnhancedPeriodSelectionResult:
    """Result from enhanced period selection."""
    
    # Selected periods
    optimal_periods: List[int]
    period_scores: Dict[int, float]  # Combined scores
    
    # Analysis results
    data_analysis_result: Optional[PeriodAnalysisResult] = None
    economic_evaluation_result: Optional[EconomicPeriodEvaluationResult] = None
    
    # Rankings
    statistical_rankings: List[Tuple[int, float]] = None
    economic_rankings: List[Tuple[int, float]] = None
    combined_rankings: List[Tuple[int, float]] = None
    
    # Summary statistics
    best_period: int = 0
    best_score: float = 0.0
    average_score: float = 0.0
    
    # Performance metrics
    total_execution_time: float = 0.0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    
    # Configuration
    config: EnhancedPeriodSelectionConfig = None
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None


class EnhancedDataDrivenPeriodSelector:
    """
    Enhanced Data-Driven Period Selector with Economic Significance Evaluation.
    
    Combines statistical analysis from DataDrivenPeriodSelector with economic
    significance evaluation and backtesting, following the pattern of
    DataDrivenInteractionGenerator.
    """
    
    def __init__(self, config: Optional[EnhancedPeriodSelectionConfig] = None):
        """
        Initialize the enhanced data-driven period selector.
        
        Args:
            config: Configuration for enhanced period selection
        """
        self.config = config or EnhancedPeriodSelectionConfig()
        self.logger = logger
        
        # Initialize data-driven period selector
        self.data_driven_selector = DataDrivenPeriodSelector(
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            max_periods=self.config.max_periods,
            min_data_points=self.config.min_data_points,
            enable_vectorbt=self.config.enable_vectorbt,
            enable_parallel=self.config.enable_parallel,
            memory_efficient=self.config.memory_efficient
        )
        
        # Initialize economic evaluator
        if self.config.enable_economic_evaluation:
            economic_config = EconomicEvaluationConfig(
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                backtest_periods=self.config.backtest_periods,
                min_backtest_periods=self.config.min_backtest_periods,
                enable_vectorbt=self.config.enable_vectorbt,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient
            )
            self.economic_evaluator = EconomicPeriodEvaluator(economic_config)
        else:
            self.economic_evaluator = None
        
        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'total_execution_time': 0.0,
            'data_analysis_operations': 0,
            'economic_evaluation_operations': 0
        }
        
        tprint_info("🚀 Enhanced Data-Driven Period Selector initialized")
        tprint_debug(f"📊 Configuration: max_periods={self.config.max_periods}, "
                    f"economic_evaluation={self.config.enable_economic_evaluation}")
    
    def select_optimal_periods(self, 
                              data: pd.DataFrame, 
                              target_timeframe: Optional[str] = None) -> EnhancedPeriodSelectionResult:
        """
        Select optimal periods using both statistical and economic analysis.
        
        Args:
            data: Input data for analysis
            target_timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            EnhancedPeriodSelectionResult with selected periods and analysis
        """
        start_time = time.time()
        
        def _validate_inputs():
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Data must be a non-empty DataFrame")
            if 'close' not in data.columns:
                raise ValueError("Data must contain 'close' column")
        
        def _select_periods():
            tprint_info("🔍 Starting enhanced period selection...")
            tprint_debug(f"📊 Data shape: {data.shape}, target timeframe: {target_timeframe}")
            
            # Step 1: Data-driven statistical analysis
            tprint_info("📈 Step 1: Statistical analysis...")
            data_analysis_result = self.data_driven_selector.select_optimal_periods(data, target_timeframe)
            
            if not data_analysis_result.optimal_periods:
                tprint_warning("⚠️ No periods found in statistical analysis")
                return self._create_empty_result(start_time, "No periods found in statistical analysis")
            
            tprint_success(f"✅ Statistical analysis found {len(data_analysis_result.optimal_periods)} periods")
            
            # Step 2: Economic significance evaluation (if enabled)
            economic_evaluation_result = None
            if self.config.enable_economic_evaluation and self.economic_evaluator:
                tprint_info("💰 Step 2: Economic significance evaluation...")
                
                # Get candidate periods from statistical analysis
                candidate_periods = data_analysis_result.optimal_periods
                
                # Evaluate economic significance
                economic_evaluation_result = self.economic_evaluator.evaluate_periods(
                    data, candidate_periods, target_timeframe or "15m"
                )
                
                if not economic_evaluation_result.top_periods:
                    tprint_warning("⚠️ No periods passed economic evaluation")
                    return self._create_empty_result(start_time, "No periods passed economic evaluation")
                
                tprint_success(f"✅ Economic evaluation completed: {economic_evaluation_result.successful_evaluations} successful")
            else:
                tprint_info("⚠️ Economic evaluation disabled, using statistical analysis only")
            
            # Step 3: Combine results
            tprint_info("🔄 Step 3: Combining statistical and economic analysis...")
            
            if economic_evaluation_result and economic_evaluation_result.top_periods:
                # Combine statistical and economic rankings
                optimal_periods, period_scores, rankings = self._combine_analysis_results(
                    data_analysis_result, economic_evaluation_result
                )
            else:
                # Use statistical analysis only
                optimal_periods = data_analysis_result.optimal_periods
                period_scores = {period: 1.0 for period in optimal_periods}
                rankings = [(period, 1.0) for period in optimal_periods]
            
            # Step 4: Filter by economic significance threshold (if applicable)
            if self.config.enable_economic_evaluation and economic_evaluation_result:
                filtered_periods = [
                    period for period in optimal_periods
                    if period_scores.get(period, 0.0) >= self.config.min_economic_score
                ]
                
                if filtered_periods:
                    optimal_periods = filtered_periods
                    tprint_success(f"✅ Filtered to {len(optimal_periods)} economically significant periods")
                else:
                    tprint_warning("⚠️ No periods met economic significance threshold, using top statistical periods")
            
            # Step 5: Limit to max_periods
            if len(optimal_periods) > self.config.max_periods:
                optimal_periods = optimal_periods[:self.config.max_periods]
                tprint_info(f"📊 Limited to top {self.config.max_periods} periods")
            
            # Calculate summary statistics
            best_period = optimal_periods[0] if optimal_periods else 0
            best_score = period_scores.get(best_period, 0.0)
            average_score = np.mean([period_scores.get(p, 0.0) for p in optimal_periods]) if optimal_periods else 0.0
            
            total_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_selections': 1,
                'successful_selections': 1,
                'total_execution_time': total_time,
                'data_analysis_operations': 1,
                'economic_evaluation_operations': 1 if economic_evaluation_result else 0
            })
            
            tprint_success(f"✅ Enhanced period selection completed in {total_time:.3f}s")
            tprint_info(f"🏆 Selected {len(optimal_periods)} optimal periods: {optimal_periods}")
            tprint_info(f"📊 Best period: {best_period} (score: {best_score:.3f})")
            
            return EnhancedPeriodSelectionResult(
                optimal_periods=optimal_periods,
                period_scores=period_scores,
                data_analysis_result=data_analysis_result,
                economic_evaluation_result=economic_evaluation_result,
                statistical_rankings=self._create_statistical_rankings(data_analysis_result),
                economic_rankings=self._create_economic_rankings(economic_evaluation_result),
                combined_rankings=rankings,
                best_period=best_period,
                best_score=best_score,
                average_score=average_score,
                total_execution_time=total_time,
                successful_evaluations=len(optimal_periods),
                failed_evaluations=0,
                config=self.config,
                success=True
            )
        
        # Execute with error handling
        try:
            _validate_inputs()
            return _select_periods()
        except Exception as e:
            tprint_error(f"❌ Enhanced period selection failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _combine_analysis_results(self, 
                                 data_analysis_result: PeriodAnalysisResult,
                                 economic_evaluation_result: EconomicPeriodEvaluationResult) -> Tuple[List[int], Dict[int, float], List[Tuple[int, float]]]:
        """Combine statistical and economic analysis results."""
        try:
            # Get all periods from both analyses
            all_periods = set(data_analysis_result.optimal_periods)
            if economic_evaluation_result.top_periods:
                all_periods.update(economic_evaluation_result.top_periods)
            
            # Create normalized scores
            statistical_scores = self._normalize_statistical_scores(data_analysis_result)
            economic_scores = self._normalize_economic_scores(economic_evaluation_result)
            
            # Combine scores with weights
            combined_scores = {}
            for period in all_periods:
                stat_score = statistical_scores.get(period, 0.0)
                econ_score = economic_scores.get(period, 0.0)
                
                combined_score = (
                    stat_score * self.config.statistical_weight +
                    econ_score * self.config.economic_weight
                )
                combined_scores[period] = combined_score
            
            # Create rankings
            rankings = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select optimal periods
            optimal_periods = [period for period, _ in rankings]
            
            return optimal_periods, combined_scores, rankings
            
        except Exception as e:
            self.logger.error(f"Failed to combine analysis results: {e}")
            # Fallback to statistical analysis only
            return data_analysis_result.optimal_periods, {}, []
    
    def _normalize_statistical_scores(self, data_analysis_result: PeriodAnalysisResult) -> Dict[int, float]:
        """Normalize statistical analysis scores to 0-1 range."""
        try:
            if not data_analysis_result.optimal_periods:
                return {}
            
            # Simple normalization - all periods get equal weight from statistical analysis
            # In a more sophisticated implementation, you could use confidence scores
            max_periods = len(data_analysis_result.optimal_periods)
            scores = {}
            
            for i, period in enumerate(data_analysis_result.optimal_periods):
                # Higher rank = higher score
                scores[period] = (max_periods - i) / max_periods
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Failed to normalize statistical scores: {e}")
            return {}
    
    def _normalize_economic_scores(self, economic_evaluation_result: EconomicPeriodEvaluationResult) -> Dict[int, float]:
        """Normalize economic evaluation scores to 0-1 range."""
        try:
            if not economic_evaluation_result or not economic_evaluation_result.period_rankings:
                return {}
            
            # Use economic scores directly (already 0-1)
            scores = {}
            for period, score in economic_evaluation_result.period_rankings:
                scores[period] = score
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Failed to normalize economic scores: {e}")
            return {}
    
    def _create_statistical_rankings(self, data_analysis_result: PeriodAnalysisResult) -> List[Tuple[int, float]]:
        """Create statistical rankings."""
        if not data_analysis_result.optimal_periods:
            return []
        
        max_periods = len(data_analysis_result.optimal_periods)
        return [(period, (max_periods - i) / max_periods) 
                for i, period in enumerate(data_analysis_result.optimal_periods)]
    
    def _create_economic_rankings(self, economic_evaluation_result: EconomicPeriodEvaluationResult) -> List[Tuple[int, float]]:
        """Create economic rankings."""
        if not economic_evaluation_result or not economic_evaluation_result.period_rankings:
            return []
        
        return economic_evaluation_result.period_rankings
    
    def _create_empty_result(self, start_time: float, error_message: str = None) -> EnhancedPeriodSelectionResult:
        """Create empty result for failed selection."""
        return EnhancedPeriodSelectionResult(
            optimal_periods=[],
            period_scores={},
            total_execution_time=time.time() - start_time,
            successful_evaluations=0,
            failed_evaluations=1,
            config=self.config,
            success=False,
            error_message=error_message
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component stats
        if hasattr(self.data_driven_selector, 'get_performance_stats'):
            stats['data_driven_selector'] = self.data_driven_selector.get_performance_stats()
        
        if self.economic_evaluator and hasattr(self.economic_evaluator, 'get_performance_stats'):
            stats['economic_evaluator'] = self.economic_evaluator.get_performance_stats()
        
        return stats


# Convenience functions
def select_enhanced_periods(data: pd.DataFrame,
                           target_timeframe: str = "15m",
                           config: Optional[EnhancedPeriodSelectionConfig] = None) -> EnhancedPeriodSelectionResult:
    """
    Convenience function to select periods using enhanced analysis.
    
    Args:
        data: Input data for analysis
        target_timeframe: Target timeframe
        config: Optional configuration
        
    Returns:
        EnhancedPeriodSelectionResult with selected periods
    """
    selector = EnhancedDataDrivenPeriodSelector(config)
    return selector.select_optimal_periods(data, target_timeframe)


def get_economically_optimized_periods(data: pd.DataFrame,
                                      target_timeframe: str = "15m",
                                      min_economic_score: float = 0.4) -> List[int]:
    """
    Get periods optimized for economic significance.
    
    Args:
        data: Input data for analysis
        target_timeframe: Target timeframe
        min_economic_score: Minimum economic score threshold
        
    Returns:
        List of economically optimized periods
    """
    config = EnhancedPeriodSelectionConfig(
        enable_economic_evaluation=True,
        min_economic_score=min_economic_score
    )
    
    result = select_enhanced_periods(data, target_timeframe, config)
    return result.optimal_periods


# Export main classes and functions
__all__ = [
    'EnhancedDataDrivenPeriodSelector',
    'EnhancedPeriodSelectionConfig',
    'EnhancedPeriodSelectionResult',
    'select_enhanced_periods',
    'get_economically_optimized_periods'
]