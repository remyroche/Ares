"""
Optimized Cross Timeframe Analysis Integration

This module integrates all the optimized components and provides the main interface
for optimized cross timeframe analysis.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import system_logger
from .optimized_cross_timeframe_analysis import (
    OptimizedCrossTimeframeAnalysis,
    OptimizedCrossTimeframeConfig,
    OptimizedCrossTimeframeResult
)
from .optimized_cross_timeframe_analysis_methods import OptimizedCrossTimeframeMethods
from .optimized_cross_timeframe_analysis_advanced import OptimizedCrossTimeframeAdvanced

logger = system_logger.getChild('OptimizedCrossTimeframeIntegration')

class OptimizedCrossTimeframeAnalysisPipeline:
    """
    Complete optimized cross timeframe analysis pipeline.
    
    This class integrates all optimization components and provides a unified interface
    for performing highly optimized cross timeframe analysis.
    """
    
    def __init__(self, config: Optional[OptimizedCrossTimeframeConfig] = None):
        """Initialize the optimized cross timeframe analysis pipeline."""
        self.config = config or OptimizedCrossTimeframeConfig()
        self.logger = logger.getChild('OptimizedPipeline')
        
        # Initialize main analyzer
        self.analyzer = OptimizedCrossTimeframeAnalysis(self.config)
        
        # Initialize method classes
        self.methods = OptimizedCrossTimeframeMethods(self.analyzer)
        self.advanced = OptimizedCrossTimeframeAdvanced(self.analyzer)
        
        # Integrate methods into analyzer
        self._integrate_methods()
        
        self.logger.info("✅ Optimized Cross Timeframe Analysis Pipeline initialized")
    
    def _integrate_methods(self):
        """Integrate method classes into the main analyzer."""
        # Data loading and validation methods
        self.analyzer._load_and_validate_data = self.methods._load_and_validate_data
        self.analyzer._load_single_timeframe_data = self.methods._load_single_timeframe_data
        self.analyzer._resample_data_optimized = self.methods._resample_data_optimized
        self.analyzer._chunked_resample = self.methods._chunked_resample
        self.analyzer._optimize_dataframe_memory = self.methods._optimize_dataframe_memory
        
        # Timeframe alignment methods
        self.analyzer._align_timeframes_optimized = self.methods._align_timeframes_optimized
        self.analyzer._align_single_timeframe = self.methods._align_single_timeframe
        
        # Feature engineering methods
        self.analyzer._engineer_features_optimized = self.methods._engineer_features_optimized
        self.analyzer._create_base_features_gpu_accelerated = self.methods._create_base_features_gpu_accelerated
        self.analyzer._create_base_features_cpu = self.methods._create_base_features_cpu
        self.analyzer._create_interaction_features_parallel = self.methods._create_interaction_features_parallel
        self.analyzer._create_correlation_features = self.methods._create_correlation_features
        self.analyzer._create_momentum_features = self.methods._create_momentum_features
        self.analyzer._create_volatility_features = self.methods._create_volatility_features
        self.analyzer._create_volume_features = self.methods._create_volume_features
        self.analyzer._create_aggregation_features_parallel = self.methods._create_aggregation_features_parallel
        self.analyzer._create_timeframe_aggregation_features = self.methods._create_timeframe_aggregation_features
        self.analyzer._create_specialized_features_parallel = self.methods._create_specialized_features_parallel
        self.analyzer._create_microstructure_features = self.methods._create_microstructure_features
        self.analyzer._create_order_flow_features = self.methods._create_order_flow_features
        self.analyzer._create_momentum_divergence_features = self.methods._create_momentum_divergence_features
        self.analyzer._create_volatility_spillover_features = self.methods._create_volatility_spillover_features
        
        # Advanced methods
        self.analyzer._perform_advanced_feature_selection = self.advanced._perform_advanced_feature_selection
        self.analyzer._calculate_interaction_metrics_optimized = self.advanced._calculate_interaction_metrics_optimized
        self.analyzer._calculate_timeframe_correlations_optimized = self.advanced._calculate_timeframe_correlations_optimized
        self.analyzer._calculate_feature_importance_optimized = self.advanced._calculate_feature_importance_optimized
        self.analyzer._calculate_financial_risk_metrics = self.advanced._calculate_financial_risk_metrics
        self.analyzer._generate_quality_report = self.advanced._generate_quality_report
    
    async def analyze_cross_timeframes(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: Optional[List[str]] = None
    ) -> OptimizedCrossTimeframeResult:
        """
        Perform optimized cross timeframe analysis.
        
        Args:
            data_dir: Data directory path
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to analyze (optional)
            
        Returns:
            OptimizedCrossTimeframeResult with comprehensive analysis results
        """
        return await self.analyzer.analyze_cross_timeframes(data_dir, symbol, exchange, timeframes)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of available optimizations."""
        return {
            'hardware_optimizations': {
                'm1_memory_optimizer': self.analyzer.memory_optimizer is not None,
                'm1_cpu_optimizer': self.analyzer.cpu_optimizer is not None,
                'm1_gpu_manager': self.analyzer.gpu_manager is not None
            },
            'feature_selection': {
                'advanced_feature_selector': self.analyzer.feature_selector is not None
            },
            'utilities': {
                'data_validator': self.analyzer.data_validator is not None,
                'data_cleaner': self.analyzer.data_cleaner is not None,
                'data_transformer': self.analyzer.data_transformer is not None,
                'parquet_utils': self.analyzer.parquet_utils is not None,
                'json_serializer': self.analyzer.json_serializer is not None,
                'parquet_serializer': self.analyzer.parquet_serializer is not None
            },
            'caching': {
                'intelligent_cache': self.analyzer.cache is not None
            },
            'config': {
                'enable_m1_optimizations': self.config.enable_m1_optimizations,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                'enable_advanced_feature_selection': self.config.enable_advanced_feature_selection,
                'enable_caching': self.config.enable_caching,
                'memory_limit_gb': self.config.memory_limit_gb,
                'max_workers': self.config.max_workers
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the last analysis."""
        if hasattr(self.analyzer, '_last_result') and self.analyzer._last_result:
            return self.analyzer._last_result.performance_metrics
        return {}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if self.analyzer.memory_optimizer:
            return self.analyzer.memory_optimizer.get_memory_report()
        return {}
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        if self.analyzer.memory_optimizer:
            return self.analyzer.memory_optimizer.optimize_memory()
        return {}

# Convenience functions for easy integration
async def analyze_cross_timeframes_optimized(
    data_dir: str,
    symbol: str,
    exchange: str,
    timeframes: Optional[List[str]] = None,
    config: Optional[OptimizedCrossTimeframeConfig] = None
) -> OptimizedCrossTimeframeResult:
    """
    Convenience function to perform optimized cross timeframe analysis.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        timeframes: List of timeframes to analyze (optional)
        config: Configuration for analysis (optional)
        
    Returns:
        OptimizedCrossTimeframeResult with comprehensive analysis results
    """
    pipeline = OptimizedCrossTimeframeAnalysisPipeline(config)
    return await pipeline.analyze_cross_timeframes(data_dir, symbol, exchange, timeframes)

def create_optimized_config(
    timeframes: Optional[List[str]] = None,
    enable_m1_optimizations: bool = True,
    enable_gpu_acceleration: bool = True,
    enable_advanced_feature_selection: bool = True,
    memory_limit_gb: float = 8.0,
    max_workers: int = 4,
    **kwargs
) -> OptimizedCrossTimeframeConfig:
    """
    Create an optimized configuration for cross timeframe analysis.
    
    Args:
        timeframes: List of timeframes to analyze
        enable_m1_optimizations: Enable M1 hardware optimizations
        enable_gpu_acceleration: Enable GPU acceleration
        enable_advanced_feature_selection: Enable advanced feature selection
        memory_limit_gb: Memory limit in GB
        max_workers: Maximum number of workers for parallel processing
        **kwargs: Additional configuration parameters
        
    Returns:
        OptimizedCrossTimeframeConfig instance
    """
    config_dict = {
        'timeframes': timeframes or ['1m', '5m', '15m', '30m'],
        'enable_m1_optimizations': enable_m1_optimizations,
        'enable_gpu_acceleration': enable_gpu_acceleration,
        'enable_advanced_feature_selection': enable_advanced_feature_selection,
        'memory_limit_gb': memory_limit_gb,
        'max_workers': max_workers,
        **kwargs
    }
    
    return OptimizedCrossTimeframeConfig(**config_dict)

# Example usage and testing
async def example_usage():
    """Example usage of the optimized cross timeframe analysis."""
    try:
        # Create optimized configuration
        config = create_optimized_config(
            timeframes=['1m', '5m', '15m', '30m'],
            enable_m1_optimizations=True,
            enable_gpu_acceleration=True,
            enable_advanced_feature_selection=True,
            memory_limit_gb=8.0,
            max_workers=4
        )
        
        # Create pipeline
        pipeline = OptimizedCrossTimeframeAnalysisPipeline(config)
        
        # Check optimization status
        status = pipeline.get_optimization_status()
        print("Optimization Status:", status)
        
        # Perform analysis
        result = await pipeline.analyze_cross_timeframes(
            data_dir="data/training",
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframes=['1m', '5m', '15m', '30m']
        )
        
        # Print results
        print(f"Analysis completed successfully!")
        print(f"Features generated: {len(result.cross_timeframe_features.columns)}")
        print(f"Selected features: {len(result.selected_features.get('final', []))}")
        print(f"Performance metrics: {result.performance_metrics}")
        
        return result
        
    except Exception as e:
        print(f"Example usage failed: {e}")
        return None

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())