"""
Feature Lookback Optimization Component.

This component optimizes feature lookback periods for better model performance.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class FeatureLookbackOptimizationComponent(BaseMarketAnalysisComponent):
    """
    Feature Lookback Optimization Component.
    
    Optimizes feature lookback periods for better model performance.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the feature lookback optimization component."""
        super().__init__(config)
        self.logger = system_logger.getChild('FeatureLookbackOptimization')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['feature_lookback_optimization_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute feature lookback optimization.
        
        Args:
            data: Market data for feature optimization
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with feature lookback optimization results
        """
        self.logger.info('⚙️ Starting Feature Lookback Optimization')
        
        try:
            # Import feature optimization utilities
            from src.feature_engineering.feature_generation_optimization import get_feature_optimizer, FeatureOptimizationConfig
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for feature lookback optimization")
            
            # Get labeled data from previous stage
            triple_barrier_labeling = pipeline_state.get('triple_barrier_labeling_result', {})
            if not triple_barrier_labeling:
                raise ValueError("No triple barrier labeling results available for feature optimization")
            
            # Configure feature optimization
            optimization_config = FeatureOptimizationConfig(
                optimization_method='genetic_algorithm',
                lookback_range=(5, 50),  # 5 to 50 periods
                feature_types=['technical_indicators', 'price_features', 'volume_features'],
                optimization_metric='sharpe_ratio',
                cross_validation_folds=5,
                test_size=0.2,
                random_state=42,
                
                # Genetic algorithm parameters
                population_size=50,
                generations=100,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elitism_rate=0.1,
                
                # Feature selection
                enable_feature_selection=True,
                max_features=20,
                feature_importance_threshold=0.01,
                
                # Regime-aware optimization
                enable_regime_aware_optimization=True,
                regime_specific_optimization=True,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Get feature optimizer
            feature_optimizer = get_feature_optimizer(optimization_config)
            
            # Perform feature lookback optimization
            optimization_result = await self._perform_feature_optimization(
                feature_optimizer, market_data, triple_barrier_labeling, optimization_config
            )
            
            # Extract results
            optimization_results = optimization_result.get('optimization_results', {})
            optimized_features = optimization_result.get('optimized_features', {})
            optimization_metrics = optimization_result.get('optimization_metrics', {})
            
            # Validate that we have optimization results
            if not optimization_results or not optimized_features:
                raise ValueError("Feature lookback optimization completed but no optimization results were created")
            
            # Create single consolidated artifact
            artifacts = {
                'feature_lookback_optimization_result': {
                    'optimization_results': optimization_results,
                    'optimized_features': optimized_features,
                    'optimization_metrics': optimization_metrics,
                    'optimization_summary': {
                        'best_lookback_period': optimization_results.get('best_lookback_period', 0),
                        'best_score': optimization_results.get('best_score', 0.0),
                        'total_features_optimized': len(optimized_features),
                        'optimization_time': optimization_result.get('optimization_time', 0.0)
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            self.logger.info(f'✅ Feature Lookback Optimization completed: {len(optimized_features)} features optimized')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'features_optimized': len(optimized_features)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ Feature Lookback Optimization failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for feature optimization."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_feature_optimization(
        self, 
        feature_optimizer: Any, 
        market_data: Any, 
        triple_barrier_labeling: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual feature optimization process."""
        try:
            # Prepare data for optimization
            prepared_data = self._prepare_data_for_optimization(market_data, triple_barrier_labeling)
            
            # Perform feature optimization
            optimization_result = await feature_optimizer.optimize_features(prepared_data, config)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Feature optimization process failed: {e}")
            # Return fallback optimization result
            return {
                'optimization_results': {
                    'best_lookback_period': 20,
                    'best_score': 0.0,
                    'optimization_method': 'fallback',
                    'error': str(e)
                },
                'optimized_features': {},
                'optimization_metrics': {
                    'optimization_method': 'fallback',
                    'error': str(e)
                },
                'optimization_time': 0.0
            }
    
    def _prepare_data_for_optimization(self, data: Any, triple_barrier_labeling: Dict[str, Any]) -> Any:
        """Prepare market data and labeled data for optimization."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'triple_barrier_labeling': triple_barrier_labeling
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for optimization: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'triple_barrier_labeling': triple_barrier_labeling
        }