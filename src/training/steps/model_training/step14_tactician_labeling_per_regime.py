"""Step 14: Tactician Labeling - Per-Regime Implementation.

This module provides per-HMM regime tactician labeling functionality, ensuring that
tactician labels are created specifically for each regime's characteristics and market behavior.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step14_tactician_labeling import Step14TacticianLabeling
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


logger = get_logger('Step14TacticianLabelingPerRegime')


class PerRegimeTacticianLabelingStep(Step14TacticianLabeling):
    """Tactician labeling step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_tactician_labeling', True)
        self.regime_specific_configs = config.get('regime_specific_tactician_configs', {})
        self.adaptive_tactician_strategies = config.get('adaptive_tactician_strategies_per_regime', True)
        
    @traced(span_name='execute_per_regime_tactician_labeling')
    @per_regime_step('step14_tactician_labeling')
    async def execute_per_regime_tactician_labeling(
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
        """Execute tactician labeling on a per-regime basis.
        
        Each regime may require different tactician labeling strategies, so tactician
        labels should be created specifically for each regime's market behavior.
        
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
            self.logger.info(f"🚀 Starting per-regime tactician labeling for regime {regime_id}")
            
            # Load analyst ensemble creation results from previous step
            ensemble_data = await self._load_analyst_ensemble_data(symbol, exchange, timeframe, data_dir, regime_id)
            if ensemble_data is None:
                self.logger.error(f"❌ Failed to load analyst ensemble data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_tactician_config(regime_id)
            
            # Apply regime-specific tactician labeling
            labeling_results = await self._apply_regime_tactician_labeling(
                ensemble_data, regime_config, regime_id
            )
            
            if labeling_results is None:
                self.logger.error(f"❌ Failed tactician labeling for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_labeling_results(
                labeling_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed tactician labeling for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save labeling results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime tactician labeling for regime {regime_id}: {e}")
            return False
    
    async def _load_analyst_ensemble_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load analyst ensemble creation data for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            Analyst ensemble creation data or None
        """
        try:
            # Try per-regime analyst ensemble data first
            ensemble_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_analyst_ensemble_creation_regime_{regime_id}.json'
            
            if not ensemble_path.exists():
                # Fall back to aggregated analyst ensemble data
                ensemble_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_analyst_ensemble_creation_aggregated.json'
            
            if ensemble_path.exists():
                with open(ensemble_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"✅ Loaded analyst ensemble data for regime {regime_id}")
                return data
            else:
                self.logger.error(f"❌ Analyst ensemble data not found: {ensemble_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading analyst ensemble data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_tactician_config(self, regime_id: int) -> Dict[str, Any]:
        """Get tactician labeling configuration for a specific regime.
        
        Different regimes may require different tactician labeling strategies and parameters.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific tactician configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_trend_tactician': True,
            'enable_volatility_tactician': True,
            'enable_momentum_tactician': True,
            'enable_volume_tactician': True,
            'enable_risk_tactician': True,
            'enable_ensemble_tactician': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following tactician strategies
            return {
                **base_config,
                'tactician_strategy': {
                    'emphasis': 'trend_following',
                    'labeling_method': 'ensemble_based',
                    'confidence_threshold': 0.7,
                    'label_persistence': 3
                },
                'tactician_parameters': {
                    'trend_tactician': {
                        'trend_strength_threshold': 0.6,
                        'trend_continuation_probability': 0.75,
                        'trend_reversal_detection': 0.8
                    },
                    'momentum_tactician': {
                        'momentum_threshold': 0.5,
                        'momentum_persistence': 2,
                        'momentum_divergence_detection': 0.7
                    },
                    'volume_tactician': {
                        'volume_confirmation_threshold': 1.2,
                        'volume_divergence_sensitivity': 0.6
                    }
                }
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize volatility and risk management tactician strategies
            return {
                **base_config,
                'tactician_strategy': {
                    'emphasis': 'volatility_management',
                    'labeling_method': 'risk_aware',
                    'confidence_threshold': 0.8,
                    'label_persistence': 2
                },
                'tactician_parameters': {
                    'volatility_tactician': {
                        'volatility_threshold': 0.8,
                        'volatility_regime_detection': 0.9,
                        'volatility_forecasting': 0.7
                    },
                    'risk_tactician': {
                        'risk_tolerance': 'conservative',
                        'max_drawdown_threshold': 0.05,
                        'var_threshold': 0.02
                    },
                    'mean_reversion_tactician': {
                        'mean_reversion_threshold': 0.7,
                        'mean_reversion_timing': 0.8
                    }
                }
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'tactician_strategy': {
                    'emphasis': 'balanced_approach',
                    'labeling_method': 'adaptive_ensemble',
                    'confidence_threshold': 0.75,
                    'label_persistence': 2
                },
                'tactician_parameters': {
                    'balanced_tactician': {
                        'balance_threshold': 0.65,
                        'adaptive_weighting': True,
                        'multi_timeframe_analysis': True
                    },
                    'adaptive_tactician': {
                        'adaptation_rate': 0.1,
                        'regime_awareness': True,
                        'performance_feedback': True
                    },
                    'ensemble_tactician': {
                        'ensemble_method': 'weighted_voting',
                        'consensus_threshold': 0.6,
                        'diversity_requirement': 0.3
                    }
                }
            }
    
    async def _apply_regime_tactician_labeling(
        self,
        ensemble_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply tactician labeling to regime ensemble data.
        
        Args:
            ensemble_data: Analyst ensemble creation results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Labeling results or None
        """
        try:
            self.logger.info(f"🔧 Applying tactician labeling for regime {regime_id}")
            
            # Extract created ensembles
            created_ensembles = ensemble_data.get('created_ensembles', {})
            if not created_ensembles:
                self.logger.warning(f"⚠️ No ensembles found for tactician labeling in regime {regime_id}")
                return None
            
            results = {
                'regime_id': regime_id,
                'tactician_strategy': regime_config.get('tactician_strategy', {}),
                'tactician_parameters': regime_config.get('tactician_parameters', {}),
                'created_tacticians': {},
                'labeling_metrics': {},
                'labeling_metadata': {}
            }
            
            # Create trend tactician
            if regime_config.get('enable_trend_tactician', True):
                trend_tactician = await self._create_trend_tactician(
                    created_ensembles, regime_config, regime_id
                )
                if trend_tactician:
                    results['created_tacticians']['trend_tactician'] = trend_tactician
            
            # Create volatility tactician
            if regime_config.get('enable_volatility_tactician', True):
                volatility_tactician = await self._create_volatility_tactician(
                    created_ensembles, regime_config, regime_id
                )
                if volatility_tactician:
                    results['created_tacticians']['volatility_tactician'] = volatility_tactician
            
            # Create momentum tactician
            if regime_config.get('enable_momentum_tactician', True):
                momentum_tactician = await self._create_momentum_tactician(
                    created_ensembles, regime_config, regime_id
                )
                if momentum_tactician:
                    results['created_tacticians']['momentum_tactician'] = momentum_tactician
            
            # Create volume tactician
            if regime_config.get('enable_volume_tactician', True):
                volume_tactician = await self._create_volume_tactician(
                    created_ensembles, regime_config, regime_id
                )
                if volume_tactician:
                    results['created_tacticians']['volume_tactician'] = volume_tactician
            
            # Create risk tactician
            if regime_config.get('enable_risk_tactician', True):
                risk_tactician = await self._create_risk_tactician(
                    created_ensembles, regime_config, regime_id
                )
                if risk_tactician:
                    results['created_tacticians']['risk_tactician'] = risk_tactician
            
            # Create ensemble tactician
            if regime_config.get('enable_ensemble_tactician', True):
                ensemble_tactician = await self._create_ensemble_tactician(
                    created_ensembles, regime_config, regime_id
                )
                if ensemble_tactician:
                    results['created_tacticians']['ensemble_tactician'] = ensemble_tactician
            
            # Calculate labeling metrics
            results['labeling_metrics'] = self._calculate_labeling_metrics(results['created_tacticians'])
            
            self.logger.info(f"✅ Completed tactician labeling for regime {regime_id}: {len(results['created_tacticians'])} tacticians created")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying tactician labeling for regime {regime_id}: {e}")
            return None
    
    async def _create_trend_tactician(
        self,
        created_ensembles: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create trend tactician for regime.
        
        Args:
            created_ensembles: Created ensemble data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Trend tactician or None
        """
        try:
            tactician_params = regime_config.get('tactician_parameters', {}).get('trend_tactician', {})
            
            # Create trend tactician
            trend_tactician = {
                'tactician_type': 'trend_tactician',
                'regime_id': regime_id,
                'specialization': 'trend_analysis',
                'ensemble_integration': {
                    'primary_ensemble': 'weighted_ensemble',
                    'secondary_ensemble': 'voting_ensemble',
                    'ensemble_weights': {
                        'weighted_ensemble': 0.6,
                        'voting_ensemble': 0.4
                    }
                },
                'tactician_capabilities': {
                    'trend_detection': True,
                    'trend_continuation_analysis': True,
                    'trend_reversal_detection': True,
                    'trend_strength_measurement': True
                },
                'tactician_parameters': {
                    'trend_strength_threshold': tactician_params.get('trend_strength_threshold', 0.6),
                    'trend_continuation_probability': tactician_params.get('trend_continuation_probability', 0.75),
                    'trend_reversal_detection': tactician_params.get('trend_reversal_detection', 0.8),
                    'label_confidence_threshold': regime_config.get('tactician_strategy', {}).get('confidence_threshold', 0.7)
                },
                'labeling_strategy': {
                    'labeling_method': regime_config.get('tactician_strategy', {}).get('labeling_method', 'ensemble_based'),
                    'label_persistence': regime_config.get('tactician_strategy', {}).get('label_persistence', 3),
                    'confidence_weighting': True
                },
                'performance_metrics': {
                    'trend_accuracy': 0.0,  # Will be calculated during training
                    'trend_precision': 0.0,
                    'trend_recall': 0.0,
                    'label_consistency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created trend tactician for regime {regime_id}")
            return trend_tactician
            
        except Exception as e:
            self.logger.error(f"❌ Error creating trend tactician for regime {regime_id}: {e}")
            return None
    
    async def _create_volatility_tactician(
        self,
        created_ensembles: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create volatility tactician for regime.
        
        Args:
            created_ensembles: Created ensemble data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Volatility tactician or None
        """
        try:
            tactician_params = regime_config.get('tactician_parameters', {}).get('volatility_tactician', {})
            
            # Create volatility tactician
            volatility_tactician = {
                'tactician_type': 'volatility_tactician',
                'regime_id': regime_id,
                'specialization': 'volatility_analysis',
                'ensemble_integration': {
                    'primary_ensemble': 'stacked_ensemble',
                    'secondary_ensemble': 'dynamic_ensemble',
                    'ensemble_weights': {
                        'stacked_ensemble': 0.7,
                        'dynamic_ensemble': 0.3
                    }
                },
                'tactician_capabilities': {
                    'volatility_detection': True,
                    'volatility_regime_detection': True,
                    'volatility_forecasting': True,
                    'volatility_risk_assessment': True
                },
                'tactician_parameters': {
                    'volatility_threshold': tactician_params.get('volatility_threshold', 0.8),
                    'volatility_regime_detection': tactician_params.get('volatility_regime_detection', 0.9),
                    'volatility_forecasting': tactician_params.get('volatility_forecasting', 0.7),
                    'label_confidence_threshold': regime_config.get('tactician_strategy', {}).get('confidence_threshold', 0.8)
                },
                'labeling_strategy': {
                    'labeling_method': regime_config.get('tactician_strategy', {}).get('labeling_method', 'risk_aware'),
                    'label_persistence': regime_config.get('tactician_strategy', {}).get('label_persistence', 2),
                    'volatility_aware_labeling': True
                },
                'performance_metrics': {
                    'volatility_accuracy': 0.0,
                    'volatility_precision': 0.0,
                    'volatility_recall': 0.0,
                    'label_consistency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created volatility tactician for regime {regime_id}")
            return volatility_tactician
            
        except Exception as e:
            self.logger.error(f"❌ Error creating volatility tactician for regime {regime_id}: {e}")
            return None
    
    async def _create_momentum_tactician(
        self,
        created_ensembles: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create momentum tactician for regime.
        
        Args:
            created_ensembles: Created ensemble data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Momentum tactician or None
        """
        try:
            tactician_params = regime_config.get('tactician_parameters', {}).get('momentum_tactician', {})
            
            # Create momentum tactician
            momentum_tactician = {
                'tactician_type': 'momentum_tactician',
                'regime_id': regime_id,
                'specialization': 'momentum_analysis',
                'ensemble_integration': {
                    'primary_ensemble': 'weighted_ensemble',
                    'secondary_ensemble': 'boosting_ensemble',
                    'ensemble_weights': {
                        'weighted_ensemble': 0.5,
                        'boosting_ensemble': 0.5
                    }
                },
                'tactician_capabilities': {
                    'momentum_detection': True,
                    'momentum_strength_measurement': True,
                    'momentum_divergence_analysis': True,
                    'momentum_continuation_prediction': True
                },
                'tactician_parameters': {
                    'momentum_threshold': tactician_params.get('momentum_threshold', 0.5),
                    'momentum_persistence': tactician_params.get('momentum_persistence', 2),
                    'momentum_divergence_detection': tactician_params.get('momentum_divergence_detection', 0.7),
                    'label_confidence_threshold': regime_config.get('tactician_strategy', {}).get('confidence_threshold', 0.7)
                },
                'labeling_strategy': {
                    'labeling_method': regime_config.get('tactician_strategy', {}).get('labeling_method', 'ensemble_based'),
                    'label_persistence': regime_config.get('tactician_strategy', {}).get('label_persistence', 3),
                    'momentum_aware_labeling': True
                },
                'performance_metrics': {
                    'momentum_accuracy': 0.0,
                    'momentum_precision': 0.0,
                    'momentum_recall': 0.0,
                    'label_consistency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created momentum tactician for regime {regime_id}")
            return momentum_tactician
            
        except Exception as e:
            self.logger.error(f"❌ Error creating momentum tactician for regime {regime_id}: {e}")
            return None
    
    async def _create_volume_tactician(
        self,
        created_ensembles: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create volume tactician for regime.
        
        Args:
            created_ensembles: Created ensemble data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Volume tactician or None
        """
        try:
            tactician_params = regime_config.get('tactician_parameters', {}).get('volume_tactician', {})
            
            # Create volume tactician
            volume_tactician = {
                'tactician_type': 'volume_tactician',
                'regime_id': regime_id,
                'specialization': 'volume_analysis',
                'ensemble_integration': {
                    'primary_ensemble': 'voting_ensemble',
                    'secondary_ensemble': 'bagging_ensemble',
                    'ensemble_weights': {
                        'voting_ensemble': 0.6,
                        'bagging_ensemble': 0.4
                    }
                },
                'tactician_capabilities': {
                    'volume_analysis': True,
                    'volume_confirmation': True,
                    'volume_divergence_detection': True,
                    'volume_profile_analysis': True
                },
                'tactician_parameters': {
                    'volume_confirmation_threshold': tactician_params.get('volume_confirmation_threshold', 1.2),
                    'volume_divergence_sensitivity': tactician_params.get('volume_divergence_sensitivity', 0.6),
                    'label_confidence_threshold': regime_config.get('tactician_strategy', {}).get('confidence_threshold', 0.7)
                },
                'labeling_strategy': {
                    'labeling_method': regime_config.get('tactician_strategy', {}).get('labeling_method', 'ensemble_based'),
                    'label_persistence': regime_config.get('tactician_strategy', {}).get('label_persistence', 3),
                    'volume_aware_labeling': True
                },
                'performance_metrics': {
                    'volume_accuracy': 0.0,
                    'volume_precision': 0.0,
                    'volume_recall': 0.0,
                    'label_consistency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created volume tactician for regime {regime_id}")
            return volume_tactician
            
        except Exception as e:
            self.logger.error(f"❌ Error creating volume tactician for regime {regime_id}: {e}")
            return None
    
    async def _create_risk_tactician(
        self,
        created_ensembles: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create risk tactician for regime.
        
        Args:
            created_ensembles: Created ensemble data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Risk tactician or None
        """
        try:
            tactician_params = regime_config.get('tactician_parameters', {}).get('risk_tactician', {})
            
            # Create risk tactician
            risk_tactician = {
                'tactician_type': 'risk_tactician',
                'regime_id': regime_id,
                'specialization': 'risk_analysis',
                'ensemble_integration': {
                    'primary_ensemble': 'dynamic_ensemble',
                    'secondary_ensemble': 'stacked_ensemble',
                    'ensemble_weights': {
                        'dynamic_ensemble': 0.7,
                        'stacked_ensemble': 0.3
                    }
                },
                'tactician_capabilities': {
                    'risk_assessment': True,
                    'risk_monitoring': True,
                    'risk_control': True,
                    'risk_reporting': True
                },
                'tactician_parameters': {
                    'risk_tolerance': tactician_params.get('risk_tolerance', 'conservative'),
                    'max_drawdown_threshold': tactician_params.get('max_drawdown_threshold', 0.05),
                    'var_threshold': tactician_params.get('var_threshold', 0.02),
                    'label_confidence_threshold': regime_config.get('tactician_strategy', {}).get('confidence_threshold', 0.8)
                },
                'labeling_strategy': {
                    'labeling_method': regime_config.get('tactician_strategy', {}).get('labeling_method', 'risk_aware'),
                    'label_persistence': regime_config.get('tactician_strategy', {}).get('label_persistence', 2),
                    'risk_aware_labeling': True
                },
                'performance_metrics': {
                    'risk_accuracy': 0.0,
                    'risk_precision': 0.0,
                    'risk_recall': 0.0,
                    'label_consistency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created risk tactician for regime {regime_id}")
            return risk_tactician
            
        except Exception as e:
            self.logger.error(f"❌ Error creating risk tactician for regime {regime_id}: {e}")
            return None
    
    async def _create_ensemble_tactician(
        self,
        created_ensembles: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create ensemble tactician for regime.
        
        Args:
            created_ensembles: Created ensemble data
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Ensemble tactician or None
        """
        try:
            tactician_params = regime_config.get('tactician_parameters', {}).get('ensemble_tactician', {})
            
            # Create ensemble tactician
            ensemble_tactician = {
                'tactician_type': 'ensemble_tactician',
                'regime_id': regime_id,
                'specialization': 'ensemble_analysis',
                'ensemble_integration': {
                    'all_ensembles': list(created_ensembles.keys()),
                    'ensemble_weights': self._calculate_ensemble_weights(created_ensembles, regime_id)
                },
                'tactician_capabilities': {
                    'ensemble_prediction': True,
                    'consensus_analysis': True,
                    'confidence_weighting': True,
                    'diversity_management': True
                },
                'tactician_parameters': {
                    'ensemble_method': tactician_params.get('ensemble_method', 'weighted_voting'),
                    'consensus_threshold': tactician_params.get('consensus_threshold', 0.6),
                    'diversity_requirement': tactician_params.get('diversity_requirement', 0.3),
                    'label_confidence_threshold': regime_config.get('tactician_strategy', {}).get('confidence_threshold', 0.75)
                },
                'labeling_strategy': {
                    'labeling_method': regime_config.get('tactician_strategy', {}).get('labeling_method', 'adaptive_ensemble'),
                    'label_persistence': regime_config.get('tactician_strategy', {}).get('label_persistence', 2),
                    'ensemble_aware_labeling': True
                },
                'performance_metrics': {
                    'ensemble_accuracy': 0.0,
                    'consensus_accuracy': 0.0,
                    'ensemble_diversity': 0.0,
                    'label_consistency': 0.0
                }
            }
            
            self.logger.info(f"✅ Created ensemble tactician for regime {regime_id}")
            return ensemble_tactician
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble tactician for regime {regime_id}: {e}")
            return None
    
    def _calculate_ensemble_weights(self, created_ensembles: Dict[str, Any], regime_id: int) -> Dict[str, float]:
        """Calculate ensemble weights for ensemble tactician.
        
        Args:
            created_ensembles: Created ensemble data
            regime_id: Regime ID
            
        Returns:
            Dictionary of ensemble weights
        """
        try:
            weights = {}
            total_ensembles = len(created_ensembles)
            
            if total_ensembles == 0:
                return weights
            
            # Base weight for each ensemble
            base_weight = 1.0 / total_ensembles
            
            # Adjust weights based on regime characteristics
            for ensemble_name, ensemble_data in created_ensembles.items():
                ensemble_method = ensemble_data.get('ensemble_method', 'unknown')
                
                # Regime-specific weight adjustments
                if regime_id <= 2:  # Trending regimes
                    if ensemble_method in ['weighted_voting', 'voting']:
                        weight_multiplier = 1.2
                    else:
                        weight_multiplier = 0.8
                elif regime_id >= 5:  # Volatile regimes
                    if ensemble_method in ['stacking', 'dynamic_selection']:
                        weight_multiplier = 1.2
                    else:
                        weight_multiplier = 0.8
                else:  # Balanced regimes
                    weight_multiplier = 1.0
                
                weights[ensemble_name] = base_weight * weight_multiplier
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ensemble weights: {e}")
            return {name: 1.0 / len(created_ensembles) for name in created_ensembles.keys()}
    
    def _calculate_labeling_metrics(self, created_tacticians: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate labeling metrics.
        
        Args:
            created_tacticians: Created tactician results
            
        Returns:
            Labeling metrics
        """
        try:
            metrics = {
                'total_tacticians': len(created_tacticians),
                'tactician_types': list(created_tacticians.keys()),
                'ensemble_integration': {},
                'labeling_capabilities': {},
                'overall_labeling_performance': 0.0
            }
            
            # Analyze ensemble integration
            ensemble_usage = {}
            for tactician_name, tactician_data in created_tacticians.items():
                ensemble_integration = tactician_data.get('ensemble_integration', {})
                for ensemble_name in ensemble_integration.get('all_ensembles', []):
                    ensemble_usage[ensemble_name] = ensemble_usage.get(ensemble_name, 0) + 1
            
            metrics['ensemble_integration'] = ensemble_usage
            
            # Analyze labeling capabilities
            capabilities = set()
            for tactician_data in created_tacticians.values():
                tactician_capabilities = tactician_data.get('tactician_capabilities', {})
                capabilities.update(tactician_capabilities.keys())
            
            metrics['labeling_capabilities'] = list(capabilities)
            
            # Calculate overall performance (placeholder)
            metrics['overall_labeling_performance'] = 0.75  # Placeholder value
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating labeling metrics: {e}")
            return {'overall_labeling_performance': 0.0}
    
    async def _save_regime_labeling_results(
        self,
        labeling_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save tactician labeling results for a specific regime.
        
        Args:
            labeling_results: Labeling results
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
            labeling_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_tactician_labeling_regime_{regime_id}.json'
            
            with open(labeling_path, 'w') as f:
                json.dump(labeling_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved tactician labeling results for regime {regime_id}: {labeling_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving tactician labeling results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_tactician_labeling_step')
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
    """Run the enhanced per-regime tactician labeling step.
    
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
    logger.info("🚀 Starting Step 14: Per-Regime Tactician Labeling")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_tactician_labeling'] = True
    
    # Initialize and run the per-regime tactician labeling step
    step = PerRegimeTacticianLabelingStep(config)
    
    success = await step.execute_per_regime_tactician_labeling(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 14: Per-Regime Tactician Labeling completed successfully")
    else:
        logger.error("❌ Step 14: Per-Regime Tactician Labeling failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime tactician labeling step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime tactician labeling result: {success}')
        
    asyncio.run(test())