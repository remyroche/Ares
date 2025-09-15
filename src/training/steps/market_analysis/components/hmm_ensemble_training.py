"""
HMM Ensemble Training Component.

This component trains HMM ensemble (meta-model) with hyperparameter optimization.
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


class HMMEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    HMM Ensemble Training Component.
    
    Trains HMM ensemble (meta-model) with hyperparameter optimization.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM ensemble training component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMEnsembleTraining')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM ensemble training.
        
        Args:
            data: Market data for training
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with ensemble training results
        """
        self.logger.info('🎭 Starting HMM Ensemble Training')
        
        try:
            # Import HMM ensemble training utilities
            from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for HMM ensemble training")
            
            # Get base models from previous stage
            hmm_models_training = pipeline_state.get('hmm_models_training_result', {})
            if not hmm_models_training:
                raise ValueError("No HMM base models available for ensemble training")
            
            # Configure HMM ensemble training
            ensemble_config = {
                'ensemble_methods': ['voting', 'stacking', 'bagging'],
                'meta_models': ['random_forest', 'gradient_boosting', 'neural_network'],
                'cross_validation_folds': 5,
                'test_size': 0.2,
                'random_state': 42,
                
                # Hyperparameter optimization
                'enable_hpo': True,
                'hpo_method': 'bayesian_optimization',
                'n_trials': 30,
                'optimization_metric': 'accuracy',
                
                # Hardware optimization
                'enable_parallel_processing': True,
                'enable_gpu_acceleration': True,
                'memory_limit_gb': 8.0
            }
            
            # Create HMM composite manager
            hmm_manager = EnhancedHMMCompositeManager()
            
            # Perform HMM ensemble training
            ensemble_result = await self._perform_ensemble_training(
                hmm_manager, market_data, hmm_models_training, ensemble_config
            )
            
            # Extract results
            hmm_ensemble_models = ensemble_result.get('hmm_ensemble_models', [])
            ensemble_metrics = ensemble_result.get('ensemble_metrics', {})
            hpo_results = ensemble_result.get('hpo_results', {})
            
            # Validate that we have ensemble results
            if not hmm_ensemble_models:
                raise ValueError("HMM ensemble training completed but no ensemble models were trained")
            
            # Create single consolidated artifact
            artifacts = {
                'hmm_ensemble_training_result': {
                    'hmm_ensemble_models': hmm_ensemble_models,
                    'ensemble_metrics': ensemble_metrics,
                    'hpo_results': hpo_results,
                    'ensemble_summary': {
                        'total_ensemble_models': len(hmm_ensemble_models),
                        'best_ensemble_method': ensemble_metrics.get('best_ensemble_method', 'unknown'),
                        'best_accuracy': ensemble_metrics.get('best_accuracy', 0.0),
                        'ensemble_training_time': ensemble_result.get('ensemble_training_time', 0.0),
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
            
            self.logger.info(f'✅ HMM Ensemble Training completed: {len(hmm_ensemble_models)} ensemble models trained')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'ensemble_models_trained': len(hmm_ensemble_models)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ HMM Ensemble Training failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for ensemble training."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_ensemble_training(
        self, 
        hmm_manager: Any, 
        market_data: Any, 
        hmm_models_training: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual HMM ensemble training process."""
        try:
            # Prepare data for ensemble training
            prepared_data = self._prepare_data_for_ensemble_training(market_data, hmm_models_training)
            
            # Perform HMM ensemble training with HPO
            ensemble_result = await hmm_manager.train_hmm_ensemble(prepared_data, config)
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"HMM ensemble training process failed: {e}")
            # Return fallback ensemble result
            return {
                'hmm_ensemble_models': [],
                'ensemble_metrics': {
                    'ensemble_method': 'fallback',
                    'error': str(e)
                },
                'hpo_results': {
                    'n_trials': 0,
                    'best_score': 0.0
                },
                'ensemble_training_time': 0.0
            }
    
    def _prepare_data_for_ensemble_training(self, data: Any, hmm_models_training: Dict[str, Any]) -> Any:
        """Prepare market data and base models for ensemble training."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'hmm_models_training': hmm_models_training
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for ensemble training: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'hmm_models_training': hmm_models_training
        }