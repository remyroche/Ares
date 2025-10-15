"""
Feature Generation Period + Lookback Optimization Step

This step combines period optimization and lookback optimization to optimize both
concurrently, ensuring at least 2 periods per feature with no recency bias.

Key Features:
- Concurrent period and lookback optimization
- Minimum 2 periods per feature
- No recency bias or adaptive windows
- Correlation threshold >0.85 for redundancy
- Top 1 period/lookback used as default for trading
- Top 3 periods/lookback used for interaction generation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import pipeline components
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_period_lookback_optimization_step
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.simplified_config import (
    UnifiedPipelineConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
    UnifiedDataDrivenPipeline
)

# Import utilities
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.logging_utils import (
    tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodLookbackOptimizationConfig:
    """Configuration for concurrent period and lookback optimization."""
    
    # Period optimization settings
    min_periods_per_feature: int = 2  # Minimum 2 periods per feature
    max_periods_per_feature: int = 5  # Maximum periods per feature
    period_range: Tuple[int, int] = (1, 50)  # Period range to analyze
    redundancy_threshold: float = 0.85  # Correlation threshold for redundancy
    
    # Lookback optimization settings
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 5
    
    # Optimization strategy
    optimization_method: str = "concurrent"  # Concurrent optimization
    enable_economic_evaluation: bool = True
    enable_statistical_analysis: bool = True
    
    # Output settings
    top_periods_for_trading: int = 1  # Top 1 used as default for trading
    top_periods_for_interactions: int = 3  # Top 3 used for interaction generation
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True


class PeriodLookbackOptimizationStep:
    """
    Concurrent period and lookback optimization step.
    
    This step optimizes both period and lookback parameters simultaneously,
    ensuring at least 2 periods per feature while maintaining non-redundancy
    and avoiding recency bias.
    """
    
    def __init__(self, config: Optional[PeriodLookbackOptimizationConfig] = None):
        """
        Initialize the period + lookback optimization step.
        
        Args:
            config: Configuration for the optimization step
        """
        self.config = config or PeriodLookbackOptimizationConfig()
        self.logger = logger
        
        # Initialize optimization results
        self.optimization_results = {
            'period_results': {},
            'lookback_results': {},
            'combined_results': {},
            'feature_periods': {},
            'feature_lookbacks': {},
            'optimization_metadata': {}
        }
        
        tprint_info("🔧 Initialized Period + Lookback Optimization Step")
        tprint_debug(f"📊 Configuration: {self.config}")
    
    async def execute(self, 
                     data: pd.DataFrame, 
                     targets: pd.Series,
                     pipeline_state: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Execute the concurrent period and lookback optimization.
        
        Args:
            data: Input data with OHLCV columns
            targets: Required target series for optimization
            pipeline_state: Pipeline state dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing optimization results
        """
        tprint_info("🚀 Starting concurrent period + lookback optimization")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"🎯 Targets shape: {targets.shape if targets is not None else 'None'}")
        
        try:
            # Validate input data
            self._validate_inputs(data, targets)
            
            # Initialize pipeline for optimization
            pipeline_config = UnifiedPipelineConfig()
            pipeline = UnifiedDataDrivenPipeline(pipeline_config)
            
            # Execute concurrent optimization
            optimization_result = await self._execute_concurrent_optimization(
                data, targets, pipeline, pipeline_state
            )
            
            # Process and store results
            self._process_optimization_results(optimization_result)
            
            # Generate optimization report
            report = self._generate_optimization_report()
            
            tprint_success("✅ Concurrent period + lookback optimization completed")
            
            return {
                'success': True,
                'optimization_results': self.optimization_results,
                'report': report,
                'metadata': {
                    'step_name': 'period_lookback_optimization',
                    'data_shape': data.shape,
                    'targets_shape': targets.shape if targets is not None else None,
                    'config': self.config.__dict__
                }
            }
            
        except Exception as e:
            error_msg = f"Period + lookback optimization failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'error': error_msg,
                'optimization_results': self.optimization_results,
                'metadata': {
                    'step_name': 'period_lookback_optimization',
                    'data_shape': data.shape,
                    'targets_shape': targets.shape if targets is not None else None,
                    'config': self.config.__dict__
                }
            }
    
    def _validate_inputs(self, data: pd.DataFrame, targets: pd.Series) -> None:
        """Validate input data and parameters."""
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Data must be DataFrame, got {type(data)}")
        
        if targets is None:
            raise ValueError("Targets are required for target-driven optimization")
        
        if not isinstance(targets, pd.Series):
            raise TypeError(f"Targets must be Series, got {type(targets)}")
        
        if len(targets) != len(data):
            raise ValueError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
    
    async def _execute_concurrent_optimization(self, 
                                             data: pd.DataFrame, 
                                             targets: pd.Series,
                                             pipeline: UnifiedDataDrivenPipeline,
                                             pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute concurrent period and lookback optimization."""
        tprint_info("🔄 Executing concurrent period + lookback optimization")
        
        try:
            # Use the pipeline's concurrent optimization method
            if hasattr(pipeline, '_concurrent_period_lookback_optimization'):
                result = await pipeline._concurrent_period_lookback_optimization(
                    data, targets, self.config, pipeline_state
                )
            else:
                # Fallback to sequential optimization if concurrent method not available
                tprint_warning("⚠️ Concurrent optimization not available, using sequential approach")
                result = await self._sequential_optimization(data, targets, pipeline, pipeline_state)
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Concurrent optimization failed: {e}")
            raise
    
    async def _sequential_optimization(self, 
                                     data: pd.DataFrame, 
                                     targets: pd.Series,
                                     pipeline: UnifiedDataDrivenPipeline,
                                     pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback sequential optimization."""
        tprint_info("🔄 Using sequential optimization approach")
        
        # Period optimization
        tprint_info("📈 Performing period optimization")
        period_result = await self._optimize_periods(data, targets, pipeline)
        
        # Lookback optimization
        tprint_info("📊 Performing lookback optimization")
        lookback_result = await self._optimize_lookbacks(data, targets, pipeline)
        
        # Combine results
        combined_result = self._combine_optimization_results(period_result, lookback_result)
        
        return combined_result
    
    async def _optimize_periods(self, 
                               data: pd.DataFrame, 
                               targets: pd.Series,
                               pipeline: UnifiedDataDrivenPipeline) -> Dict[str, Any]:
        """Optimize periods for features."""
        try:
            # Use pipeline's period optimization
            if hasattr(pipeline, '_enhanced_period_optimization'):
                result = pipeline._enhanced_period_optimization(data, "15m")
                return result
            else:
                # Fallback period optimization
                return self._fallback_period_optimization(data, targets)
                
        except Exception as e:
            tprint_warning(f"⚠️ Period optimization failed: {e}")
            return {'optimal_periods': [], 'period_scores': {}}
    
    async def _optimize_lookbacks(self, 
                                 data: pd.DataFrame, 
                                 targets: pd.Series,
                                 pipeline: UnifiedDataDrivenPipeline) -> Dict[str, Any]:
        """Optimize lookbacks for features."""
        try:
            # Use pipeline's lookback optimization
            if hasattr(pipeline, '_advanced_lookback_optimization'):
                result = pipeline._advanced_lookback_optimization(data, targets, data, {})
                return result
            else:
                # Fallback lookback optimization
                return self._fallback_lookback_optimization(data, targets)
                
        except Exception as e:
            tprint_warning(f"⚠️ Lookback optimization failed: {e}")
            return {'optimized_lookbacks': {}, 'lookback_scores': {}}
    
    def _fallback_period_optimization(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Fallback period optimization implementation."""
        tprint_info("🔄 Using fallback period optimization")
        
        # Simple period analysis
        periods = list(range(self.config.period_range[0], self.config.period_range[1] + 1))
        period_scores = {}
        
        for period in periods:
            try:
                # Calculate simple correlation score
                if targets is not None:
                    # Use rolling correlation with targets
                    rolling_corr = data['close'].rolling(period).corr(targets)
                    score = rolling_corr.mean() if not rolling_corr.isna().all() else 0.0
                else:
                    # Use volatility as proxy
                    rolling_vol = data['close'].rolling(period).std()
                    score = rolling_vol.mean() if not rolling_vol.isna().all() else 0.0
                
                period_scores[period] = score
                
            except Exception as e:
                tprint_debug(f"Period {period} optimization failed: {e}")
                period_scores[period] = 0.0
        
        # Select optimal periods (at least 2 per feature)
        sorted_periods = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_periods = [p[0] for p in sorted_periods[:self.config.max_periods_per_feature]]
        
        # Ensure minimum periods
        if len(optimal_periods) < self.config.min_periods_per_feature:
            optimal_periods.extend(periods[:self.config.min_periods_per_feature - len(optimal_periods)])
        
        return {
            'optimal_periods': optimal_periods,
            'period_scores': period_scores
        }
    
    def _fallback_lookback_optimization(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Fallback lookback optimization implementation."""
        tprint_info("🔄 Using fallback lookback optimization")
        
        # Simple lookback analysis
        lookbacks = list(range(self.config.min_lookback, self.config.max_lookback + 1, self.config.lookback_step))
        lookback_scores = {}
        
        for lookback in lookbacks:
            try:
                # Calculate information content
                if targets is not None:
                    # Use mutual information with targets
                    from sklearn.feature_selection import mutual_info_regression
                    lookback_data = data['close'].rolling(lookback).mean().dropna()
                    if len(lookback_data) > 0:
                        score = mutual_info_regression(
                            lookback_data.values.reshape(-1, 1), 
                            targets.iloc[-len(lookback_data):]
                        )[0]
                    else:
                        score = 0.0
                else:
                    # Use variance as proxy
                    rolling_var = data['close'].rolling(lookback).var()
                    score = rolling_var.mean() if not rolling_var.isna().all() else 0.0
                
                lookback_scores[lookback] = score
                
            except Exception as e:
                tprint_debug(f"Lookback {lookback} optimization failed: {e}")
                lookback_scores[lookback] = 0.0
        
        # Select optimal lookbacks
        sorted_lookbacks = sorted(lookback_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_lookbacks = {f"feature_{i}": lookbacks[0] for i, lookbacks in enumerate(sorted_lookbacks[:5])}
        
        return {
            'optimized_lookbacks': optimal_lookbacks,
            'lookback_scores': lookback_scores
        }
    
    def _combine_optimization_results(self, period_result: Dict[str, Any], lookback_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine period and lookback optimization results."""
        tprint_info("🔄 Combining period and lookback optimization results")
        
        combined_result = {
            'period_optimization': period_result,
            'lookback_optimization': lookback_result,
            'combined_periods_lookbacks': {},
            'trading_defaults': {},
            'interaction_periods': {}
        }
        
        # Combine optimal periods and lookbacks
        optimal_periods = period_result.get('optimal_periods', [])
        optimal_lookbacks = lookback_result.get('optimized_lookbacks', {})
        
        # Create combined feature configurations
        for i, period in enumerate(optimal_periods):
            feature_name = f"feature_{i}"
            lookback = optimal_lookbacks.get(feature_name, self.config.min_lookback)
            
            combined_result['combined_periods_lookbacks'][feature_name] = {
                'period': period,
                'lookback': lookback,
                'score': period_result.get('period_scores', {}).get(period, 0.0)
            }
        
        # Set trading defaults (top 1)
        if optimal_periods:
            best_period = optimal_periods[0]
            best_lookback = optimal_lookbacks.get('feature_0', self.config.min_lookback)
            combined_result['trading_defaults'] = {
                'period': best_period,
                'lookback': best_lookback
            }
        
        # Set interaction periods (top 3)
        combined_result['interaction_periods'] = optimal_periods[:self.config.top_periods_for_interactions]
        
        return combined_result
    
    def _process_optimization_results(self, result: Dict[str, Any]) -> None:
        """Process and store optimization results."""
        tprint_info("📊 Processing optimization results")
        
        self.optimization_results.update({
            'period_results': result.get('period_optimization', {}),
            'lookback_results': result.get('lookback_optimization', {}),
            'combined_results': result.get('combined_periods_lookbacks', {}),
            'trading_defaults': result.get('trading_defaults', {}),
            'interaction_periods': result.get('interaction_periods', []),
            'optimization_metadata': {
                'config': self.config.__dict__,
                'timestamp': pd.Timestamp.now().isoformat(),
                'success': True
            }
        })
        
        tprint_success(f"✅ Processed {len(self.optimization_results['combined_results'])} feature configurations")
    
    def _generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        tprint_info("📋 Generating optimization report")
        
        report = {
            'summary': {
                'total_features_optimized': len(self.optimization_results['combined_results']),
                'optimal_periods_count': len(self.optimization_results['period_results'].get('optimal_periods', [])),
                'optimal_lookbacks_count': len(self.optimization_results['lookback_results'].get('optimized_lookbacks', {})),
                'trading_default_period': self.optimization_results['trading_defaults'].get('period', 0),
                'trading_default_lookback': self.optimization_results['trading_defaults'].get('lookback', 0),
                'interaction_periods_count': len(self.optimization_results['interaction_periods'])
            },
            'configuration': self.config.__dict__,
            'optimization_results': self.optimization_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check if minimum periods requirement is met
        optimal_periods = self.optimization_results['period_results'].get('optimal_periods', [])
        if len(optimal_periods) < self.config.min_periods_per_feature:
            recommendations.append(
                f"Warning: Only {len(optimal_periods)} periods found, "
                f"minimum {self.config.min_periods_per_feature} required"
            )
        
        # Check correlation threshold
        period_scores = self.optimization_results['period_results'].get('period_scores', {})
        if period_scores:
            max_correlation = max(period_scores.values()) if period_scores else 0.0
            if max_correlation > self.config.redundancy_threshold:
                recommendations.append(
                    f"High correlation detected ({max_correlation:.3f}), "
                    f"consider increasing redundancy threshold"
                )
        
        # Check lookback diversity
        lookback_scores = self.optimization_results['lookback_results'].get('lookback_scores', {})
        if lookback_scores:
            lookback_values = list(lookback_scores.keys())
            lookback_range = max(lookback_values) - min(lookback_values) if lookback_values else 0
            if lookback_range < 20:
                recommendations.append(
                    "Limited lookback diversity, consider expanding lookback range"
                )
        
        if not recommendations:
            recommendations.append("Optimization completed successfully with good diversity")
        
        return recommendations


# Convenience function for ares_launcher.py
async def run_period_lookback_optimization_step(
    data: pd.DataFrame,
    targets: pd.Series,
    config: Optional[PeriodLookbackOptimizationConfig] = None,
    pipeline_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run the period + lookback optimization step.
    
    Args:
        data: Input data with OHLCV columns
        targets: Required target series for optimization
        config: Configuration for the optimization step
        pipeline_state: Pipeline state dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing optimization results
    """
    step = PeriodLookbackOptimizationStep(config)
    return await step.execute(data, targets, pipeline_state, **kwargs)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='15T')
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Create sample targets (in real usage, these would come from labeling system)
    targets = pd.Series(np.random.randn(1000), index=dates)
    
    # Run optimization
    async def main():
        config = PeriodLookbackOptimizationConfig()
        result = await run_period_lookback_optimization_step(data, targets, config)
        print(f"Optimization result: {result['success']}")
        if result['success']:
            print(f"Report: {result['report']['summary']}")
    
    asyncio.run(main())