"""
Hybrid NAS-TAS Integration for Data-Driven Model Selection

This module integrates the data-driven model selector with the hybrid NAS-TAS regime detection system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime

from .data_driven_model_selector import DataDrivenModelSelector, ModelSelectorConfig
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector
from ..config.hybrid_regime_config import HybridRegimeConfig

logger = logging.getLogger(__name__)


class HybridModelSelector:
    """
    Hybrid NAS-TAS model selector that integrates with the hybrid regime detection system.
    """
    
    def __init__(self, 
                 hybrid_config: HybridRegimeConfig,
                 selector_config: Optional[ModelSelectorConfig] = None):
        """Initialize hybrid model selector."""
        self.hybrid_config = hybrid_config
        self.selector_config = selector_config or ModelSelectorConfig()
        
        # Initialize components
        self.hybrid_detector = HybridNASTASRegimeDetector(hybrid_config)
        self.model_selector = DataDrivenModelSelector(self.selector_config)
        
        # Hybrid-specific model registry
        self.hybrid_models = {
            # NAS models
            'neural_ode': 'Neural ODE Regime Detector',
            'vision_transformer': 'Vision Transformer Regime Detector', 
            'neural_state_space': 'Neural State Space Model',
            'hybrid_architecture': 'Hybrid Regime Architecture',
            'few_shot_learner': 'Few-Shot Regime Learner',
            'continual_learner': 'Continual Learning Model',
            'uncertainty_estimator': 'Uncertainty Estimator',
            
            # TAS models
            'tree_based_clustering': 'Tree-Based Clustering',
            'statistical_validation': 'Statistical Validation',
            'clvsa_enhanced': 'CLVSA Enhanced Detection',
            'hybrid_detection': 'Hybrid Detection Method',
            'meta_learning_adapted': 'Meta-Learning Adapted',
            'bootstrap_validated': 'Bootstrap Validated',
            'multi_timeframe': 'Multi-Timeframe Detection',
            'real_time_streaming': 'Real-Time Streaming Detection',
            
            # Hybrid models
            'nas_tas_ensemble': 'NAS-TAS Ensemble',
            'economic_clustering': 'Economic Clustering',
            'coherent_modeling': 'Coherent Regime Modeling',
            'momentum_analysis': 'Momentum Analysis',
            'volume_profile': 'Volume Profile Analysis'
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("✅ Hybrid Model Selector initialized")
    
    def detect_regimes_and_select_models(self,
                                       market_data: Union[pd.DataFrame, np.ndarray],
                                       timestamps: Optional[np.ndarray] = None,
                                       available_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect regimes and select optimal models for each regime using hybrid approach.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            available_models: List of available hybrid models
            
        Returns:
            Dictionary with regime detection and model selection results
        """
        try:
            start_time = time.time()
            
            # Step 1: Detect regimes using hybrid NAS-TAS
            self.logger.info("🔀 Detecting regimes using hybrid NAS-TAS system...")
            hybrid_result = self.hybrid_detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                validate_economic_significance=True,
                validate_financial_relevance=True
            )
            
            if not hybrid_result.success:
                raise ValueError(f"Hybrid regime detection failed: {hybrid_result.error_message}")
            
            # Step 2: Get available models
            if available_models is None:
                available_models = list(self.hybrid_models.keys())
            
            # Step 3: Select models for each detected regime
            regime_model_selections = {}
            unique_regimes = np.unique(hybrid_result.regime_predictions)
            
            for regime_id in unique_regimes:
                self.logger.info(f"🎯 Selecting models for regime {regime_id}...")
                
                # Get regime-specific data
                regime_mask = hybrid_result.regime_predictions == regime_id
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
                        regime_data, hybrid_result, regime_id
                    ),
                    'regime_confidence': hybrid_result.regime_stability_scores[regime_mask].mean() if len(hybrid_result.regime_stability_scores) > 0 else 0.5,
                    'economic_significance': hybrid_result.economic_significance_scores[regime_mask].mean() if len(hybrid_result.economic_significance_scores) > 0 else 0.5,
                    'financial_relevance': hybrid_result.financial_relevance_scores[regime_mask].mean() if len(hybrid_result.financial_relevance_scores) > 0 else 0.5
                }
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            result = {
                'success': True,
                'execution_time': execution_time,
                'hybrid_result': hybrid_result,
                'regime_model_selections': regime_model_selections,
                'system_summary': self.model_selector.get_system_summary(),
                'metadata': {
                    'system': 'Hybrid NAS-TAS Data-Driven Model Selection',
                    'n_regimes_detected': len(unique_regimes),
                    'models_available': len(available_models),
                    'ensemble_enabled': self.selector_config.enable_ensemble,
                    'combination_strategy': self.hybrid_config.combination_strategy.value,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"✅ Hybrid regime detection and model selection completed in {execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(unique_regimes)}")
            self.logger.info(f"   Models available: {len(available_models)}")
            self.logger.info(f"   Combination strategy: {self.hybrid_config.combination_strategy.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Hybrid regime detection and model selection failed: {e}")
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
            model_name: Name of the hybrid model
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
            
            self.logger.info(f"Updated performance for hybrid model {model_name} in regime {regime_id}: "
                           f"F1={metrics.f1_score:.3f}, Accuracy={metrics.accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to update hybrid model performance: {e}")
            raise
    
    def get_regime_insights(self, regime_id: int) -> Dict[str, Any]:
        """Get insights about model performance in a specific regime."""
        return self.model_selector.get_regime_insights(regime_id)
    
    def get_optimal_model_for_regime(self, regime_id: int) -> Tuple[str, Dict[str, float]]:
        """Get the optimal model for a specific regime."""
        available_models = list(self.hybrid_models.keys())
        return self.model_selector.select_model_for_regime(regime_id, available_models)
    
    def _extract_regime_characteristics(self,
                                      regime_data: np.ndarray,
                                      hybrid_result,
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
                'hybrid_confidence': 0.0
            }
            
            # Add hybrid-specific characteristics
            if hasattr(hybrid_result, 'economic_significance_scores'):
                regime_mask = hybrid_result.regime_predictions == regime_id
                if len(hybrid_result.economic_significance_scores) > 0:
                    characteristics['economic_significance'] = np.mean(hybrid_result.economic_significance_scores[regime_mask])
            
            if hasattr(hybrid_result, 'financial_relevance_scores'):
                regime_mask = hybrid_result.regime_predictions == regime_id
                if len(hybrid_result.financial_relevance_scores) > 0:
                    characteristics['financial_relevance'] = np.mean(hybrid_result.financial_relevance_scores[regime_mask])
            
            if hasattr(hybrid_result, 'regime_stability_scores'):
                regime_mask = hybrid_result.regime_predictions == regime_id
                if len(hybrid_result.regime_stability_scores) > 0:
                    characteristics['stability_score'] = np.mean(hybrid_result.regime_stability_scores[regime_mask])
            
            # Add hybrid-specific features
            if hasattr(hybrid_result, 'combined_features') and hybrid_result.combined_features is not None:
                regime_mask = hybrid_result.regime_predictions == regime_id
                if len(hybrid_result.combined_features) > 0:
                    characteristics['combined_features'] = np.mean(hybrid_result.combined_features[regime_mask])
            
            # Add TAS contributions
            if hasattr(hybrid_result, 'tas_contributions') and hybrid_result.tas_contributions:
                characteristics['tas_contributions'] = hybrid_result.tas_contributions
            
            # Add NAS contributions
            if hasattr(hybrid_result, 'nas_contributions') and hybrid_result.nas_contributions:
                characteristics['nas_contributions'] = hybrid_result.nas_contributions
            
            # Add clustering metrics
            if hasattr(hybrid_result, 'clustering_metrics') and hybrid_result.clustering_metrics:
                characteristics['clustering_metrics'] = hybrid_result.clustering_metrics
            
            # Add economic clustering metrics
            if hasattr(hybrid_result, 'economic_clustering_metrics') and hybrid_result.economic_clustering_metrics:
                characteristics['economic_clustering_metrics'] = hybrid_result.economic_clustering_metrics
            
            # Add momentum scores
            if hasattr(hybrid_result, 'momentum_scores') and hybrid_result.momentum_scores is not None:
                regime_mask = hybrid_result.regime_predictions == regime_id
                if len(hybrid_result.momentum_scores) > 0:
                    characteristics['momentum_score'] = np.mean(hybrid_result.momentum_scores[regime_mask])
            
            # Add volume profiles
            if hasattr(hybrid_result, 'volume_profiles') and hybrid_result.volume_profiles is not None:
                regime_mask = hybrid_result.regime_predictions == regime_id
                if len(hybrid_result.volume_profiles) > 0:
                    characteristics['volume_profile'] = np.mean(hybrid_result.volume_profiles[regime_mask])
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Failed to extract regime characteristics: {e}")
            return {'regime_id': regime_id, 'error': str(e)}
    
    def save_mappings(self):
        """Save current model mappings."""
        self.model_selector.save_mappings()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the hybrid model selection system."""
        summary = self.model_selector.get_system_summary()
        summary['system_type'] = 'Hybrid NAS-TAS Data-Driven Model Selection'
        summary['hybrid_models_available'] = len(self.hybrid_models)
        summary['hybrid_config'] = {
            'combination_strategy': self.hybrid_config.combination_strategy.value,
            'tas_weight': self.hybrid_config.tas_config.get('base_weight', 0.4),
            'nas_weight': self.hybrid_config.nas_config.get('base_weight', 0.6),
            'economic_evaluation': self.hybrid_config.economic_evaluation.get('enabled', True),
            'economic_clustering': self.hybrid_config.clustering_config.get('economic_clustering', True)
        }
        return summary