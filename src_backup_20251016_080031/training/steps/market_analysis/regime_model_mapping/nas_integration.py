"""
NAS Integration for Data-Driven Model Selection

This module integrates the data-driven model selector with the NAS regime detection system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)
from .data_driven_model_selector import DataDrivenModelSelector, ModelSelectorConfig
from ..nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector
from ..nas_regime.core.perfect_nas_config import PerfectNASConfig

logger = logging.getLogger(__name__)


class NASModelSelector:
    """
    NAS-specific model selector that integrates with the Perfect NAS Regime System.
    """
    
    def __init__(self, 
                 nas_config: PerfectNASConfig,
                 selector_config: Optional[ModelSelectorConfig] = None):
        """Initialize NAS model selector."""
        tprint("🚀 [NAS_INTEGRATION] Initializing NAS Model Selector", color="cyan", bold=True)
        self.nas_config = nas_config
        self.selector_config = selector_config or ModelSelectorConfig()
        
        # Initialize components
        tprint("🧠 [NAS_INTEGRATION] Initializing NAS detector", color="yellow")
        self.nas_detector = PerfectNASRegimeDetector(nas_config)
        tprint("🔧 [NAS_INTEGRATION] Initializing model selector", color="yellow")
        self.model_selector = DataDrivenModelSelector(self.selector_config)
        
        # NAS-specific model registry
        tprint("📊 [NAS_INTEGRATION] Setting up NAS model registry", color="blue")
        self.nas_models = {
            'neural_ode': 'Neural ODE Regime Detector',
            'vision_transformer': 'Vision Transformer Regime Detector', 
            'neural_state_space': 'Neural State Space Model',
            'hybrid_architecture': 'Hybrid Regime Architecture',
            'few_shot_learner': 'Few-Shot Regime Learner',
            'continual_learner': 'Continual Learning Model',
            'uncertainty_estimator': 'Uncertainty Estimator'
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint(f"✅ [NAS_INTEGRATION] NAS Model Selector initialized with {len(self.nas_models)} models", color="green")
        self.logger.info("✅ NAS Model Selector initialized")
    
    def detect_regimes_and_select_models(self,
                                       market_data: Union[pd.DataFrame, np.ndarray],
                                       timestamps: Optional[np.ndarray] = None,
                                       available_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect regimes and select optimal models for each regime.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            available_models: List of available NAS models
            
        Returns:
            Dictionary with regime detection and model selection results
        """
        try:
            start_time = time.time()
            
            # Step 1: Detect regimes using NAS
            self.logger.info("🔍 Detecting regimes using NAS system...")
            nas_result = self.nas_detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                optimize_architecture=True,
                enable_meta_learning=True
            )
            
            if not nas_result.success:
                raise ValueError(f"NAS regime detection failed: {nas_result.error_message}")
            
            # Step 2: Get available models
            if available_models is None:
                available_models = list(self.nas_models.keys())
            
            # Step 3: Select models for each detected regime
            regime_model_selections = {}
            unique_regimes = np.unique(nas_result.regime_predictions)
            
            for regime_id in unique_regimes:
                self.logger.info(f"🎯 Selecting models for regime {regime_id}...")
                
                # Get regime-specific data
                regime_mask = nas_result.regime_predictions == regime_id
                regime_data = market_data[regime_mask] if hasattr(market_data, '__getitem__') else market_data[regime_mask]
                
                # Select best model for this regime
                selected_model, ensemble_weights = self.model_selector.select_model_for_regime(
                    regime_id=int(regime_id),
                    available_models=available_models
                )
                
                # Get ensemble weights if enabled
                if self.selector_config.enable_ensemble:
                    ensemble_weights = self.model_selector.get_ensemble_weights(
                        regime_id=int(regime_id),
                        available_models=available_models
                    )
                
                regime_model_selections[regime_id] = {
                    'selected_model': selected_model,
                    'ensemble_weights': ensemble_weights,
                    'regime_characteristics': self._extract_regime_characteristics(
                        regime_data, nas_result, regime_id
                    ),
                    'regime_confidence': nas_result.regime_stability_scores[regime_mask].mean() if len(nas_result.regime_stability_scores) > 0 else 0.5
                }
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            result = {
                'success': True,
                'execution_time': execution_time,
                'nas_result': nas_result,
                'regime_model_selections': regime_model_selections,
                'system_summary': self.model_selector.get_system_summary(),
                'metadata': {
                    'system': 'NAS Data-Driven Model Selection',
                    'n_regimes_detected': len(unique_regimes),
                    'models_available': len(available_models),
                    'ensemble_enabled': self.selector_config.enable_ensemble,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"✅ NAS regime detection and model selection completed in {execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(unique_regimes)}")
            self.logger.info(f"   Models available: {len(available_models)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ NAS regime detection and model selection failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    def update_model_performance(self,
                               regime_id: int,
                               model_name: str,
                               predictions: np.ndarray,
                               actual_values: np.ndarray,
                               execution_time: float,
                               regime_characteristics: Optional[Dict[str, Any]] = None):
        """
        Update model performance for a specific regime.
        
        Args:
            regime_id: ID of the market regime
            model_name: Name of the NAS model
            predictions: Model predictions
            actual_values: Actual values
            execution_time: Time taken for inference
            regime_characteristics: Characteristics of the regime
        """
        try:
            # Update performance in the model selector
            metrics = self.model_selector.register_model_performance(
                regime_id=regime_id,
                model_name=model_name,
                predictions=predictions,
                actual_values=actual_values,
                execution_time=execution_time,
                regime_characteristics=regime_characteristics
            )
            
            self.logger.info(f"Updated performance for NAS model {model_name} in regime {regime_id}: "
                           f"F1={metrics.f1_score:.3f}, Accuracy={metrics.accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to update NAS model performance: {e}")
            raise
    
    def get_regime_insights(self, regime_id: int) -> Dict[str, Any]:
        """Get insights about model performance in a specific regime."""
        return self.model_selector.get_regime_insights(regime_id)
    
    def get_optimal_model_for_regime(self, regime_id: int) -> Tuple[str, Dict[str, float]]:
        """Get the optimal model for a specific regime."""
        available_models = list(self.nas_models.keys())
        return self.model_selector.select_model_for_regime(regime_id, available_models)
    
    def _extract_regime_characteristics(self,
                                      regime_data: np.ndarray,
                                      nas_result,
                                      regime_id: int) -> Dict[str, Any]:
        """Extract characteristics of a regime for model selection."""
        try:
            characteristics = {
                'regime_id': regime_id,
                'data_size': len(regime_data),
                'volatility': np.std(regime_data) if len(regime_data) > 0 else 0.0,
                'mean_value': np.mean(regime_data) if len(regime_data) > 0 else 0.0,
                'trend_strength': 0.0,  # Would need to calculate trend
                'complexity_score': 0.0,  # Would need to calculate complexity
                'nas_confidence': 0.0
            }
            
            # Add NAS-specific characteristics
            if hasattr(nas_result, 'economic_significance_scores'):
                regime_mask = nas_result.regime_predictions == regime_id
                if len(nas_result.economic_significance_scores) > 0:
                    characteristics['economic_significance'] = np.mean(nas_result.economic_significance_scores[regime_mask])
            
            if hasattr(nas_result, 'trading_viability_scores'):
                regime_mask = nas_result.regime_predictions == regime_id
                if len(nas_result.trading_viability_scores) > 0:
                    characteristics['trading_viability'] = np.mean(nas_result.trading_viability_scores[regime_mask])
            
            if hasattr(nas_result, 'regime_stability_scores'):
                regime_mask = nas_result.regime_predictions == regime_id
                if len(nas_result.regime_stability_scores) > 0:
                    characteristics['stability_score'] = np.mean(nas_result.regime_stability_scores[regime_mask])
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Failed to extract regime characteristics: {e}")
            return {'regime_id': regime_id, 'error': str(e)}
    
    def save_mappings(self):
        """Save current model mappings."""
        self.model_selector.save_mappings()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the NAS model selection system."""
        summary = self.model_selector.get_system_summary()
        summary['system_type'] = 'NAS Data-Driven Model Selection'
        summary['nas_models_available'] = len(self.nas_models)
        summary['nas_config'] = {
            'primary_architecture': self.nas_config.primary_architecture.value,
            'enable_neural_odes': self.nas_config.enable_neural_odes,
            'enable_vision_transformers': self.nas_config.enable_vision_transformers,
            'enable_meta_learning': self.nas_config.enable_meta_learning
        }
        return summary