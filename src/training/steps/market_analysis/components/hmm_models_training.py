"""
HMM Models Training Component.

This component trains HMM base models with hyperparameter optimization.
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


class HMMModelsTrainingComponent(BaseMarketAnalysisComponent):
    """
    HMM Models Training Component.
    
    Trains HMM base models with hyperparameter optimization.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM models training component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMModelsTraining')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_models_training_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM models training.
        
        Args:
            data: Market data for training
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with training results
        """
        self.logger.info('🤖 Starting HMM Models Training')
        
        try:
            # Import HMM training utilities
            from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for HMM models training")
            
            # Get clustering results from previous stage
            hmm_clustering = pipeline_state.get('hmm_clustering_result', {})
            if not hmm_clustering:
                raise ValueError("No HMM clustering results available for training")
            
            # Configure HMM training
            training_config = {
                'model_types': ['gaussian', 'multinomial', 'mixture'],
                'n_states_range': [2, 3, 4, 5],
                'covariance_types': ['full', 'tied', 'diag', 'spherical'],
                'max_iterations': 100,
                'convergence_tolerance': 1e-6,
                'cross_validation_folds': 5,
                
                # Hyperparameter optimization
                'enable_hpo': True,
                'hpo_method': 'bayesian_optimization',
                'n_trials': 50,
                'optimization_metric': 'aic',
                
                # Hardware optimization
                'enable_parallel_processing': True,
                'enable_gpu_acceleration': True,
                'memory_limit_gb': 8.0
            }
            
            # Create HMM composite manager
            hmm_manager = EnhancedHMMCompositeManager()
            
            # Perform HMM models training
            training_result = await self._perform_hmm_training(
                hmm_manager, market_data, hmm_clustering, training_config
            )
            
            # Extract results
            hmm_base_models = training_result.get('hmm_base_models', [])
            training_metrics = training_result.get('training_metrics', {})
            hpo_results = training_result.get('hpo_results', {})
            
            # Validate that we have training results
            if not hmm_base_models:
                raise ValueError("HMM models training completed but no models were trained")
            
            # Create single consolidated artifact
            artifacts = {
                'hmm_models_training_result': {
                    'hmm_base_models': hmm_base_models,
                    'training_metrics': training_metrics,
                    'hpo_results': hpo_results,
                    'training_summary': {
                        'total_models_trained': len(hmm_base_models),
                        'best_model_type': training_metrics.get('best_model_type', 'unknown'),
                        'best_aic_score': training_metrics.get('best_aic_score', 0.0),
                        'training_time': training_result.get('training_time', 0.0),
                        'hpo_trials': hpo_results.get('n_trials', 0)
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
            
            self.logger.info(f'✅ HMM Models Training completed: {len(hmm_base_models)} models trained')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'models_trained': len(hmm_base_models)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ HMM Models Training failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for training."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_hmm_training(
        self, 
        hmm_manager: Any, 
        market_data: Any, 
        hmm_clustering: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual HMM training process."""
        try:
            # Prepare data for training
            prepared_data = self._prepare_data_for_training(market_data, hmm_clustering)
            
            # Perform HMM training with HPO
            training_result = await hmm_manager.train_hmm_models(prepared_data, config)
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"HMM training process failed: {e}")
            # Return fallback training result
            return {
                'hmm_base_models': [],
                'training_metrics': {
                    'training_method': 'fallback',
                    'error': str(e)
                },
                'hpo_results': {
                    'n_trials': 0,
                    'best_score': 0.0
                },
                'training_time': 0.0
            }
    
    def _prepare_data_for_training(self, data: Any, hmm_clustering: Dict[str, Any]) -> Any:
        """Prepare market data and clustering results for training."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'hmm_clustering': hmm_clustering
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for training: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'hmm_clustering': hmm_clustering
        }