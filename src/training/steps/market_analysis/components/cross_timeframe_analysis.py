"""
Cross Timeframe Analysis Component.

This component performs cross timeframe interaction feature analysis.
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


class CrossTimeframeAnalysisComponent(BaseMarketAnalysisComponent):
    """
    Cross Timeframe Analysis Component.
    
    Performs cross timeframe interaction feature analysis.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the cross timeframe analysis component."""
        super().__init__(config)
        self.logger = system_logger.getChild('CrossTimeframeAnalysis')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['cross_timeframe_analysis_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute cross timeframe analysis.
        
        Args:
            data: Market data for cross timeframe analysis
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with cross timeframe analysis results
        """
        self.logger.info('🌐 Starting Cross Timeframe Analysis')
        
        try:
            # Import cross timeframe analysis utilities
            from src.feature_engineering.cross_timeframe_features import CrossTimeframeFeatureGenerator, CrossTimeframeConfig
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for cross timeframe analysis")
            
            # Get feature optimization results from previous stage
            feature_lookback_optimization = pipeline_state.get('feature_lookback_optimization_result', {})
            if not feature_lookback_optimization:
                raise ValueError("No feature lookback optimization results available for cross timeframe analysis")
            
            # Configure cross timeframe analysis
            analysis_config = CrossTimeframeConfig(
                target_timeframe=self.config.timeframe,
                analysis_timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
                feature_types=['price_momentum', 'volume_profile', 'volatility_regime', 'trend_alignment'],
                interaction_depth=2,  # 2nd order interactions
                correlation_threshold=0.7,
                significance_threshold=0.05,
                
                # Feature generation
                enable_momentum_features=True,
                enable_volume_features=True,
                enable_volatility_features=True,
                enable_trend_features=True,
                
                # Statistical analysis
                enable_correlation_analysis=True,
                enable_causality_analysis=True,
                enable_regime_analysis=True,
                
                # Feature selection
                enable_feature_selection=True,
                max_features_per_timeframe=10,
                feature_importance_threshold=0.01,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Create cross timeframe feature generator
            feature_generator = CrossTimeframeFeatureGenerator(analysis_config)
            
            # Perform cross timeframe analysis
            analysis_result = await self._perform_cross_timeframe_analysis(
                feature_generator, market_data, feature_lookback_optimization, analysis_config
            )
            
            # Extract results
            cross_timeframe_features = analysis_result.get('cross_timeframe_features', {})
            analysis_metrics = analysis_result.get('analysis_metrics', {})
            feature_interactions = analysis_result.get('feature_interactions', {})
            
            # Validate that we have analysis results
            if not cross_timeframe_features or not analysis_metrics:
                raise ValueError("Cross timeframe analysis completed but no analysis results were created")
            
            # Create single consolidated artifact
            artifacts = {
                'cross_timeframe_analysis_result': {
                    'cross_timeframe_features': cross_timeframe_features,
                    'analysis_metrics': analysis_metrics,
                    'feature_interactions': feature_interactions,
                    'analysis_summary': {
                        'total_timeframes_analyzed': len(analysis_config.analysis_timeframes),
                        'total_features_generated': len(cross_timeframe_features),
                        'significant_interactions': len(feature_interactions.get('significant_interactions', [])),
                        'analysis_time': analysis_result.get('analysis_time', 0.0)
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
            
            self.logger.info(f'✅ Cross Timeframe Analysis completed: {len(cross_timeframe_features)} features generated')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'features_generated': len(cross_timeframe_features)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ Cross Timeframe Analysis failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for cross timeframe analysis."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_cross_timeframe_analysis(
        self, 
        feature_generator: Any, 
        market_data: Any, 
        feature_lookback_optimization: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual cross timeframe analysis process."""
        try:
            # Prepare data for analysis
            prepared_data = self._prepare_data_for_analysis(market_data, feature_lookback_optimization)
            
            # Perform cross timeframe analysis
            analysis_result = await feature_generator.generate_cross_timeframe_features(prepared_data, config)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Cross timeframe analysis process failed: {e}")
            # Return fallback analysis result
            return {
                'cross_timeframe_features': {},
                'analysis_metrics': {
                    'analysis_method': 'fallback',
                    'error': str(e)
                },
                'feature_interactions': {
                    'significant_interactions': [],
                    'correlation_matrix': {}
                },
                'analysis_time': 0.0
            }
    
    def _prepare_data_for_analysis(self, data: Any, feature_lookback_optimization: Dict[str, Any]) -> Any:
        """Prepare market data and feature optimization results for analysis."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'feature_lookback_optimization': feature_lookback_optimization
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for analysis: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'feature_lookback_optimization': feature_lookback_optimization
        }