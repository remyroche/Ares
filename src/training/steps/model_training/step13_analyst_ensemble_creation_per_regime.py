"""Step 13: Analyst Ensemble Creation - Per-Regime Implementation.

This module provides per-HMM regime analyst ensemble creation functionality, ensuring that
analyst ensembles are created specifically for each regime's characteristics and market behavior.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step13_analyst_ensemble_creation import Step13AnalystEnsembleCreation
from src.training.steps.regime_handler import regime_handler
from src.training.steps.regime_processing_decorator import (
    per_regime_processing,
    aggregate_regime_results,
    RegimeProcessingContext
)
from src.training.steps.regime_continuity_decorator import per_regime_step
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors
from src.core.decorators.errors import handles_errors


logger = get_logger('Step13AnalystEnsembleCreationPerRegime')


class PerRegimeAnalystEnsembleCreationStep(Step13AnalystEnsembleCreation):
    """Analyst ensemble creation step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_analyst_ensemble_creation', True)
        self.regime_specific_configs = config.get('regime_specific_ensemble_configs', {})
        self.adaptive_ensemble_parameters = config.get('adaptive_ensemble_parameters_per_regime', True)
        
    @traced(span_name='execute_per_regime_analyst_ensemble_creation')
    @per_regime_step('step13_analyst_ensemble_creation')
    async def execute_per_regime_analyst_ensemble_creation(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False,
        regime_id: Optional[int] = None,
        regime_context: Optional[Any] = None,
        per_regime: bool = True
    ) -> bool:
        """Execute analyst ensemble creation on a per-regime basis.
        
        Each regime may require different ensemble strategies, so analyst ensembles
        should be created specifically for each regime's market behavior.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            regime_id: Regime ID (provided by decorator)
            regime_context: Regime context (provided by decorator)
            per_regime: Per-regime flag (provided by decorator)
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"🚀 Starting per-regime analyst ensemble creation for regime {regime_id}")
            
            # Load analyst enhancement results from previous step
            enhancement_data = await self._load_analyst_enhancement_data(symbol, exchange, timeframe, data_dir, regime_id)
            if enhancement_data is None:
                self.logger.error(f"❌ Failed to load analyst enhancement data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_ensemble_config(regime_id)
            
            # Apply regime-specific analyst ensemble creation
            ensemble_results = await self._apply_regime_analyst_ensemble_creation(
                enhancement_data, regime_config, regime_id
            )
            
            if ensemble_results is None:
                self.logger.error(f"❌ Failed analyst ensemble creation for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_ensemble_results(
                ensemble_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed analyst ensemble creation for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save ensemble results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime analyst ensemble creation for regime {regime_id}: {e}")
            return False
    
    async def _load_analyst_enhancement_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load analyst enhancement data for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            Analyst enhancement data or None
        """
        try:
            # Try per-regime analyst enhancement data first
            enhancement_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_analyst_enhancement_regime_{regime_id}.json'
            
            if not enhancement_path.exists():
                # Fall back to aggregated analyst enhancement data
                enhancement_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_analyst_enhancement_aggregated.json'
            
            if enhancement_path.exists():
                with open(enhancement_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"✅ Loaded analyst enhancement data for regime {regime_id}")
                return data
            else:
                self.logger.error(f"❌ Analyst enhancement data not found: {enhancement_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading analyst enhancement data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_ensemble_config(self, regime_id: int) -> Dict[str, Any]:
        """Get analyst ensemble configuration for a specific regime.
        
        Different regimes may require different ensemble strategies and parameters.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific ensemble configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_weighted_ensemble': True,
            'enable_stacked_ensemble': True,
            'enable_voting_ensemble': True,
            'enable_boosting_ensemble': True,
            'enable_bagging_ensemble': True,
            'enable_dynamic_ensemble': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following ensemble strategies
            return {
                **base_config,
                'ensemble_strategy': {
                    'emphasis': 'trend_following',
                    'ensemble_method': 'weighted_voting',
                    'diversity_requirement': 0.4,
                    'consensus_threshold': 0.7
                },
                'ensemble_parameters': {
                    'weighted_ensemble': {
                        'trend_analyst_weight': 0.4,
                        'momentum_analyst_weight': 0.3,
                        'volume_analyst_weight': 0.2,
                        'risk_analyst_weight': 0.1
                    },
                    'stacked_ensemble': {
                        'meta_learner': 'logistic_regression',
                        'cross_validation_folds': 5,
                        'stacking_levels': 2
                    },
                    'voting_ensemble': {
                        'voting_type': 'soft',
                        'confidence_threshold': 0.75,
                        'consensus_required': True
                    }
                }
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize volatility and risk management ensemble strategies
            return {
                **base_config,
                'ensemble_strategy': {
                    'emphasis': 'volatility_management',
                    'ensemble_method': 'stacked_ensemble',
                    'diversity_requirement': 0.6,
                    'consensus_threshold': 0.8
                },
                'ensemble_parameters': {
                    'weighted_ensemble': {
                        'volatility_analyst_weight': 0.35,
                        'risk_analyst_weight': 0.3,
                        'mean_reversion_analyst_weight': 0.25,
                        'volume_analyst_weight': 0.1
                    },
                    'stacked_ensemble': {
                        'meta_learner': 'random_forest',
                        'cross_validation_folds': 7,
                        'stacking_levels': 3
                    },
                    'voting_ensemble': {
                        'voting_type': 'hard',
                        'confidence_threshold': 0.85,
                        'consensus_required': True
                    }
                }
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'ensemble_strategy': {
                    'emphasis': 'balanced_ensemble',
                    'ensemble_method': 'dynamic_ensemble',
                    'diversity_requirement': 0.5,
                    'consensus_threshold': 0.75
                },
                'ensemble_parameters': {
                    'weighted_ensemble': {
                        'balanced_analyst_weight': 0.3,
                        'adaptive_analyst_weight': 0.25,
                        'ensemble_analyst_weight': 0.25,
                        'trend_analyst_weight': 0.1,
                        'volatility_analyst_weight': 0.1
                    },
                    'stacked_ensemble': {
                        'meta_learner': 'gradient_boosting',
                        'cross_validation_folds': 6,
                        'stacking_levels': 2
                    },
                    'voting_ensemble': {
                        'voting_type': 'adaptive',
                        'confidence_threshold': 0.8,
                        'consensus_required': True
                    }
                }
            }
    
    async def _apply_regime_analyst_ensemble_creation(
        self,
        enhancement_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply analyst ensemble creation to regime enhancement data.
        
        Args:
            enhancement_data: Analyst enhancement results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Ensemble creation results or None
        """
        try:
            self.logger.info(f"🔧 Applying analyst ensemble creation for regime {regime_id}")
            
            # Extract enhanced analysts
            enhanced_analysts = enhancement_data.get('enhanced_analysts', {})
            if not enhanced_analysts:
                self.logger.warning(f"⚠️ No enhanced analysts found for ensemble creation in regime {regime_id}")
                return None
            
            results = {
                'regime_id': regime_id,
                'ensemble_strategy': regime_config.get('ensemble_strategy', {}),
                'ensemble_parameters': regime_config.get('ensemble_parameters', {}),
                'created_ensembles': {},
                'ensemble_performance': {},
                'ensemble_metadata': {}
            }
            
            # Create weighted ensemble
            if regime_config.get('enable_weighted_ensemble', True):
                weighted_ensemble = await self._create_weighted_ensemble(
                    enhanced_analysts, regime_config, regime_id
                )
                if weighted_ensemble:
                    results['created_ensembles']['weighted_ensemble'] = weighted_ensemble
            
            # Create stacked ensemble
            if regime_config.get('enable_stacked_ensemble', True):
                stacked_ensemble = await self._create_stacked_ensemble(
                    enhanced_analysts, regime_config, regime_id
                )
                if stacked_ensemble:
                    results['created_ensembles']['stacked_ensemble'] = stacked_ensemble
            
            # Create voting ensemble
            if regime_config.get('enable_voting_ensemble', True):
                voting_ensemble = await self._create_voting_ensemble(
                    enhanced_analysts, regime_config, regime_id
                )
                if voting_ensemble:
                    results['created_ensembles']['voting_ensemble'] = voting_ensemble
            
            # Create boosting ensemble
            if regime_config.get('enable_boosting_ensemble', True):
                boosting_ensemble = await self._create_boosting_ensemble(
                    enhanced_analysts, regime_config, regime_id
                )
                if boosting_ensemble:
                    results['created_ensembles']['boosting_ensemble'] = boosting_ensemble
            
            # Create bagging ensemble
            if regime_config.get('enable_bagging_ensemble', True):
                bagging_ensemble = await self._create_bagging_ensemble(
                    enhanced_analysts, regime_config, regime_id
                )
                if bagging_ensemble:
                    results['created_ensembles']['bagging_ensemble'] = bagging_ensemble
            
            # Create dynamic ensemble
            if regime_config.get('enable_dynamic_ensemble', True):
                dynamic_ensemble = await self._create_dynamic_ensemble(
                    enhanced_analysts, regime_config, regime_id
                )
                if dynamic_ensemble:
                    results['created_ensembles']['dynamic_ensemble'] = dynamic_ensemble
            
            # Calculate ensemble performance
            results['ensemble_performance'] = self._calculate_ensemble_performance(results['created_ensembles'])
            
            self.logger.info(f"✅ Completed analyst ensemble creation for regime {regime_id}: {len(results['created_ensembles'])} ensembles created")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying analyst ensemble creation for regime {regime_id}: {e}")
            return None
    
    async def _create_weighted_ensemble(
        self,
        enhanced_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create weighted ensemble for regime.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Weighted ensemble or None
        """
        try:
            ensemble_params = regime_config.get('ensemble_parameters', {}).get('weighted_ensemble', {})
            
            # Calculate analyst weights based on performance and regime characteristics
            analyst_weights = self._calculate_analyst_weights(enhanced_analysts, ensemble_params, regime_id)
            
            # Create weighted ensemble
            weighted_ensemble = {
                'ensemble_type': 'weighted_ensemble',
                'regime_id': regime_id,
                'ensemble_method': 'weighted_voting',
                'analyst_weights': analyst_weights,
                'total_weight': sum(analyst_weights.values()),
                'ensemble_parameters': {
                    'weight_calculation_method': 'performance_based',
                    'regime_adaptation': True,
                    'dynamic_weighting': True
                },
                'ensemble_capabilities': {
                    'weighted_prediction': True,
                    'confidence_weighting': True,
                    'adaptive_weights': True,
                    'performance_monitoring': True
                },
                'performance_metrics': {
                    'ensemble_accuracy': 0.0,  # Will be calculated during training
                    'weighted_consensus': 0.0,
                    'ensemble_diversity': 0.0
                }
            }
            
            self.logger.info(f"✅ Created weighted ensemble for regime {regime_id}")
            return weighted_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating weighted ensemble for regime {regime_id}: {e}")
            return None
    
    def _calculate_analyst_weights(
        self,
        enhanced_analysts: Dict[str, Any],
        ensemble_params: Dict[str, Any],
        regime_id: int
    ) -> Dict[str, float]:
        """Calculate analyst weights for ensemble.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            ensemble_params: Ensemble parameters
            regime_id: Regime ID
            
        Returns:
            Dictionary of analyst weights
        """
        try:
            weights = {}
            
            # Get performance-based weights
            for analyst_name, analyst_data in enhanced_analysts.items():
                performance_metrics = analyst_data.get('enhanced_performance_metrics', {})
                
                # Calculate average performance
                performance_scores = []
                for metric_name, metric_value in performance_metrics.items():
                    if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                        performance_scores.append(metric_value)
                
                if performance_scores:
                    avg_performance = np.mean(performance_scores)
                else:
                    avg_performance = 0.5  # Default performance
                
                # Apply regime-specific weight adjustments
                base_weight = ensemble_params.get(f'{analyst_name}_weight', 0.1)
                
                # Adjust weight based on performance and regime characteristics
                if regime_id <= 2:  # Trending regimes
                    if 'trend' in analyst_name.lower() or 'momentum' in analyst_name.lower():
                        performance_multiplier = 1.2
                    else:
                        performance_multiplier = 0.8
                elif regime_id >= 5:  # Volatile regimes
                    if 'volatility' in analyst_name.lower() or 'risk' in analyst_name.lower():
                        performance_multiplier = 1.2
                    else:
                        performance_multiplier = 0.8
                else:  # Balanced regimes
                    performance_multiplier = 1.0
                
                weights[analyst_name] = base_weight * avg_performance * performance_multiplier
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating analyst weights: {e}")
            return {name: 1.0 / len(enhanced_analysts) for name in enhanced_analysts.keys()}
    
    async def _create_stacked_ensemble(
        self,
        enhanced_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create stacked ensemble for regime.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Stacked ensemble or None
        """
        try:
            ensemble_params = regime_config.get('ensemble_parameters', {}).get('stacked_ensemble', {})
            
            # Create stacked ensemble
            stacked_ensemble = {
                'ensemble_type': 'stacked_ensemble',
                'regime_id': regime_id,
                'ensemble_method': 'stacking',
                'base_analysts': list(enhanced_analysts.keys()),
                'meta_learner': ensemble_params.get('meta_learner', 'logistic_regression'),
                'ensemble_parameters': {
                    'cross_validation_folds': ensemble_params.get('cross_validation_folds', 5),
                    'stacking_levels': ensemble_params.get('stacking_levels', 2),
                    'meta_learner_params': self._get_meta_learner_params(ensemble_params.get('meta_learner', 'logistic_regression'))
                },
                'ensemble_capabilities': {
                    'stacked_prediction': True,
                    'meta_learning': True,
                    'cross_validation': True,
                    'multi_level_stacking': True
                },
                'performance_metrics': {
                    'stacking_accuracy': 0.0,
                    'meta_learner_performance': 0.0,
                    'stacking_diversity': 0.0
                }
            }
            
            self.logger.info(f"✅ Created stacked ensemble for regime {regime_id}")
            return stacked_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating stacked ensemble for regime {regime_id}: {e}")
            return None
    
    def _get_meta_learner_params(self, meta_learner: str) -> Dict[str, Any]:
        """Get meta learner parameters.
        
        Args:
            meta_learner: Meta learner type
            
        Returns:
            Meta learner parameters
        """
        meta_learner_params = {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
        }
        
        return meta_learner_params.get(meta_learner, meta_learner_params['logistic_regression'])
    
    async def _create_voting_ensemble(
        self,
        enhanced_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create voting ensemble for regime.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Voting ensemble or None
        """
        try:
            ensemble_params = regime_config.get('ensemble_parameters', {}).get('voting_ensemble', {})
            
            # Create voting ensemble
            voting_ensemble = {
                'ensemble_type': 'voting_ensemble',
                'regime_id': regime_id,
                'ensemble_method': 'voting',
                'voting_analysts': list(enhanced_analysts.keys()),
                'voting_type': ensemble_params.get('voting_type', 'soft'),
                'ensemble_parameters': {
                    'confidence_threshold': ensemble_params.get('confidence_threshold', 0.75),
                    'consensus_required': ensemble_params.get('consensus_required', True),
                    'tie_breaking_method': 'performance_based'
                },
                'ensemble_capabilities': {
                    'voting_prediction': True,
                    'consensus_analysis': True,
                    'confidence_voting': True,
                    'tie_breaking': True
                },
                'performance_metrics': {
                    'voting_accuracy': 0.0,
                    'consensus_rate': 0.0,
                    'voting_confidence': 0.0
                }
            }
            
            self.logger.info(f"✅ Created voting ensemble for regime {regime_id}")
            return voting_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating voting ensemble for regime {regime_id}: {e}")
            return None
    
    async def _create_boosting_ensemble(
        self,
        enhanced_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create boosting ensemble for regime.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Boosting ensemble or None
        """
        try:
            # Create boosting ensemble
            boosting_ensemble = {
                'ensemble_type': 'boosting_ensemble',
                'regime_id': regime_id,
                'ensemble_method': 'boosting',
                'base_analysts': list(enhanced_analysts.keys()),
                'ensemble_parameters': {
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'max_depth': 3,
                    'regime_adaptive_boosting': True
                },
                'ensemble_capabilities': {
                    'boosting_prediction': True,
                    'sequential_learning': True,
                    'error_correction': True,
                    'adaptive_boosting': True
                },
                'performance_metrics': {
                    'boosting_accuracy': 0.0,
                    'boosting_precision': 0.0,
                    'boosting_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created boosting ensemble for regime {regime_id}")
            return boosting_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating boosting ensemble for regime {regime_id}: {e}")
            return None
    
    async def _create_bagging_ensemble(
        self,
        enhanced_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create bagging ensemble for regime.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Bagging ensemble or None
        """
        try:
            # Create bagging ensemble
            bagging_ensemble = {
                'ensemble_type': 'bagging_ensemble',
                'regime_id': regime_id,
                'ensemble_method': 'bagging',
                'base_analysts': list(enhanced_analysts.keys()),
                'ensemble_parameters': {
                    'n_estimators': 100,
                    'max_samples': 0.8,
                    'max_features': 0.8,
                    'bootstrap': True,
                    'regime_adaptive_bagging': True
                },
                'ensemble_capabilities': {
                    'bagging_prediction': True,
                    'bootstrap_aggregating': True,
                    'variance_reduction': True,
                    'parallel_processing': True
                },
                'performance_metrics': {
                    'bagging_accuracy': 0.0,
                    'bagging_precision': 0.0,
                    'bagging_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created bagging ensemble for regime {regime_id}")
            return bagging_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating bagging ensemble for regime {regime_id}: {e}")
            return None
    
    async def _create_dynamic_ensemble(
        self,
        enhanced_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create dynamic ensemble for regime.
        
        Args:
            enhanced_analysts: Enhanced analyst data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Dynamic ensemble or None
        """
        try:
            # Create dynamic ensemble
            dynamic_ensemble = {
                'ensemble_type': 'dynamic_ensemble',
                'regime_id': regime_id,
                'ensemble_method': 'dynamic_selection',
                'base_analysts': list(enhanced_analysts.keys()),
                'ensemble_parameters': {
                    'selection_method': 'performance_based',
                    'adaptation_rate': 0.1,
                    'regime_awareness': True,
                    'dynamic_weighting': True
                },
                'ensemble_capabilities': {
                    'dynamic_prediction': True,
                    'adaptive_selection': True,
                    'regime_adaptation': True,
                    'performance_monitoring': True
                },
                'performance_metrics': {
                    'dynamic_accuracy': 0.0,
                    'adaptation_rate': 0.0,
                    'selection_efficiency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created dynamic ensemble for regime {regime_id}")
            return dynamic_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating dynamic ensemble for regime {regime_id}: {e}")
            return None
    
    def _calculate_ensemble_performance(self, created_ensembles: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble performance metrics.
        
        Args:
            created_ensembles: Created ensemble results
            
        Returns:
            Performance metrics
        """
        try:
            performance_metrics = {
                'total_ensembles': len(created_ensembles),
                'ensemble_types': list(created_ensembles.keys()),
                'ensemble_diversity': 0.0,
                'overall_ensemble_performance': 0.0
            }
            
            # Calculate diversity score
            ensemble_types = set(ensemble.get('ensemble_method', 'unknown') for ensemble in created_ensembles.values())
            performance_metrics['ensemble_diversity'] = len(ensemble_types) / len(created_ensembles) if created_ensembles else 0.0
            
            # Calculate overall performance (placeholder - would be calculated during training)
            performance_metrics['overall_ensemble_performance'] = 0.75  # Placeholder value
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ensemble performance: {e}")
            return {'overall_ensemble_performance': 0.0}
    
    async def _save_regime_ensemble_results(
        self,
        ensemble_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save analyst ensemble creation results for a specific regime.
        
        Args:
            ensemble_results: Ensemble creation results
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            True if successful
        """
        try:
            # Save regime-specific results
            ensemble_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_analyst_ensemble_creation_regime_{regime_id}.json'
            
            with open(ensemble_path, 'w') as f:
                json.dump(ensemble_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved analyst ensemble creation results for regime {regime_id}: {ensemble_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving analyst ensemble creation results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_analyst_ensemble_creation_step')
@validates()
@handles_errors
async def run_per_regime_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the enhanced per-regime analyst ensemble creation step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting Step 13: Per-Regime Analyst Ensemble Creation")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_analyst_ensemble_creation'] = True
    
    # Initialize and run the per-regime analyst ensemble creation step
    step = PerRegimeAnalystEnsembleCreationStep(config)
    
    success = await step.execute_per_regime_analyst_ensemble_creation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 13: Per-Regime Analyst Ensemble Creation completed successfully")
    else:
        logger.error("❌ Step 13: Per-Regime Analyst Ensemble Creation failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime analyst ensemble creation step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime analyst ensemble creation result: {success}')
        
    asyncio.run(test())