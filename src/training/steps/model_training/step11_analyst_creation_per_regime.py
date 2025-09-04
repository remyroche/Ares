"""Step 11: Analyst Creation - Per-Regime Implementation.

This module provides per-HMM regime analyst creation functionality, ensuring that
analysts are created specifically for each regime's characteristics and market behavior.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step11_analyst_creation import Step11AnalystCreation
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


logger = get_logger('Step11AnalystCreationPerRegime')


class PerRegimeAnalystCreationStep(Step11AnalystCreation):
    """Analyst creation step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_analyst_creation', True)
        self.regime_specific_configs = config.get('regime_specific_analyst_configs', {})
        self.adaptive_analyst_parameters = config.get('adaptive_analyst_parameters_per_regime', True)
        
    @traced(span_name='execute_per_regime_analyst_creation')
    @per_regime_step('step11_analyst_creation')
    async def execute_per_regime_analyst_creation(
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
        """Execute analyst creation on a per-regime basis.
        
        Each regime may require different analyst characteristics, so analysts
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
            self.logger.info(f"🚀 Starting per-regime analyst creation for regime {regime_id}")
            
            # Load regime intelligence data from previous step
            intelligence_data = await self._load_regime_intelligence_data(symbol, exchange, timeframe, data_dir, regime_id)
            if intelligence_data is None:
                self.logger.error(f"❌ Failed to load regime intelligence data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_analyst_config(regime_id)
            
            # Apply regime-specific analyst creation
            analyst_results = await self._apply_regime_analyst_creation(
                intelligence_data, regime_config, regime_id
            )
            
            if analyst_results is None:
                self.logger.error(f"❌ Failed analyst creation for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_analyst_results(
                analyst_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed analyst creation for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save analyst results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime analyst creation for regime {regime_id}: {e}")
            return False
    
    async def _load_regime_intelligence_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load regime intelligence data for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            Regime intelligence data or None
        """
        try:
            # Try per-regime intelligence data first
            intelligence_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_intelligence_regime_{regime_id}.json'
            
            if not intelligence_path.exists():
                # Fall back to aggregated intelligence data
                intelligence_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_intelligence_aggregated.json'
            
            if intelligence_path.exists():
                with open(intelligence_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"✅ Loaded regime intelligence data for regime {regime_id}")
                return data
            else:
                self.logger.error(f"❌ Regime intelligence data not found: {intelligence_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading regime intelligence data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_analyst_config(self, regime_id: int) -> Dict[str, Any]:
        """Get analyst creation configuration for a specific regime.
        
        Different regimes may require different analyst characteristics and capabilities.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific analyst configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_trend_analyst': True,
            'enable_volatility_analyst': True,
            'enable_momentum_analyst': True,
            'enable_volume_analyst': True,
            'enable_risk_analyst': True,
            'enable_ensemble_analyst': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following analysts
            return {
                **base_config,
                'analyst_specialization': {
                    'primary_focus': 'trend_analysis',
                    'secondary_focus': 'momentum_analysis',
                    'tertiary_focus': 'volume_confirmation'
                },
                'analyst_parameters': {
                    'trend_analyst': {
                        'trend_strength_threshold': 0.6,
                        'trend_persistence_required': 3,
                        'trend_confirmation_methods': ['price_action', 'volume', 'momentum']
                    },
                    'momentum_analyst': {
                        'momentum_threshold': 0.5,
                        'momentum_persistence_required': 2,
                        'momentum_indicators': ['RSI', 'MACD', 'Stochastic']
                    },
                    'volume_analyst': {
                        'volume_confirmation_required': True,
                        'volume_threshold': 1.2,
                        'volume_analysis_methods': ['OBV', 'Volume_Profile', 'Volume_Ratio']
                    }
                },
                'analyst_weights': {
                    'trend_analyst': 0.4,
                    'momentum_analyst': 0.3,
                    'volume_analyst': 0.2,
                    'risk_analyst': 0.1
                }
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize volatility and mean-reversion analysts
            return {
                **base_config,
                'analyst_specialization': {
                    'primary_focus': 'volatility_analysis',
                    'secondary_focus': 'mean_reversion',
                    'tertiary_focus': 'risk_management'
                },
                'analyst_parameters': {
                    'volatility_analyst': {
                        'volatility_threshold': 0.8,
                        'volatility_analysis_methods': ['ATR', 'Bollinger_Bands', 'Volatility_Cones'],
                        'volatility_regime_detection': True
                    },
                    'mean_reversion_analyst': {
                        'mean_reversion_threshold': 0.7,
                        'mean_reversion_indicators': ['RSI', 'Williams_R', 'CCI'],
                        'mean_reversion_confirmation': True
                    },
                    'risk_analyst': {
                        'risk_tolerance': 'conservative',
                        'max_drawdown_threshold': 0.05,
                        'var_threshold': 0.02
                    }
                },
                'analyst_weights': {
                    'volatility_analyst': 0.35,
                    'mean_reversion_analyst': 0.3,
                    'risk_analyst': 0.25,
                    'volume_analyst': 0.1
                }
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'analyst_specialization': {
                    'primary_focus': 'balanced_analysis',
                    'secondary_focus': 'adaptive_strategies',
                    'tertiary_focus': 'multi_timeframe'
                },
                'analyst_parameters': {
                    'balanced_analyst': {
                        'balance_threshold': 0.65,
                        'adaptive_weighting': True,
                        'multi_timeframe_analysis': True
                    },
                    'adaptive_analyst': {
                        'adaptation_speed': 'medium',
                        'adaptation_methods': ['regime_detection', 'performance_feedback'],
                        'adaptation_threshold': 0.6
                    },
                    'ensemble_analyst': {
                        'ensemble_method': 'weighted_voting',
                        'confidence_weighting': True,
                        'diversity_requirement': 0.3
                    }
                },
                'analyst_weights': {
                    'balanced_analyst': 0.3,
                    'adaptive_analyst': 0.25,
                    'ensemble_analyst': 0.25,
                    'trend_analyst': 0.1,
                    'volatility_analyst': 0.1
                }
            }
    
    async def _apply_regime_analyst_creation(
        self,
        intelligence_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply analyst creation to regime intelligence data.
        
        Args:
            intelligence_data: Regime intelligence results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Analyst creation results or None
        """
        try:
            self.logger.info(f"🔧 Applying analyst creation for regime {regime_id}")
            
            # Extract intelligence components
            intelligence_components = intelligence_data.get('intelligence_components', {})
            if not intelligence_components:
                self.logger.warning(f"⚠️ No intelligence components found for regime {regime_id}")
                return None
            
            results = {
                'regime_id': regime_id,
                'analyst_specialization': regime_config.get('analyst_specialization', {}),
                'analyst_parameters': regime_config.get('analyst_parameters', {}),
                'analyst_weights': regime_config.get('analyst_weights', {}),
                'created_analysts': {},
                'analyst_performance': {},
                'analyst_metadata': {}
            }
            
            # Create trend analyst
            if regime_config.get('enable_trend_analyst', True):
                trend_analyst = await self._create_trend_analyst(
                    intelligence_components, regime_config, regime_id
                )
                if trend_analyst:
                    results['created_analysts']['trend_analyst'] = trend_analyst
            
            # Create volatility analyst
            if regime_config.get('enable_volatility_analyst', True):
                volatility_analyst = await self._create_volatility_analyst(
                    intelligence_components, regime_config, regime_id
                )
                if volatility_analyst:
                    results['created_analysts']['volatility_analyst'] = volatility_analyst
            
            # Create momentum analyst
            if regime_config.get('enable_momentum_analyst', True):
                momentum_analyst = await self._create_momentum_analyst(
                    intelligence_components, regime_config, regime_id
                )
                if momentum_analyst:
                    results['created_analysts']['momentum_analyst'] = momentum_analyst
            
            # Create volume analyst
            if regime_config.get('enable_volume_analyst', True):
                volume_analyst = await self._create_volume_analyst(
                    intelligence_components, regime_config, regime_id
                )
                if volume_analyst:
                    results['created_analysts']['volume_analyst'] = volume_analyst
            
            # Create risk analyst
            if regime_config.get('enable_risk_analyst', True):
                risk_analyst = await self._create_risk_analyst(
                    intelligence_components, regime_config, regime_id
                )
                if risk_analyst:
                    results['created_analysts']['risk_analyst'] = risk_analyst
            
            # Create ensemble analyst
            if regime_config.get('enable_ensemble_analyst', True):
                ensemble_analyst = await self._create_ensemble_analyst(
                    results['created_analysts'], regime_config, regime_id
                )
                if ensemble_analyst:
                    results['created_analysts']['ensemble_analyst'] = ensemble_analyst
            
            # Calculate analyst performance
            results['analyst_performance'] = self._calculate_analyst_performance(results['created_analysts'])
            
            self.logger.info(f"✅ Completed analyst creation for regime {regime_id}: {len(results['created_analysts'])} analysts created")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying analyst creation for regime {regime_id}: {e}")
            return None
    
    async def _create_trend_analyst(
        self,
        intelligence_components: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create trend analyst for regime.
        
        Args:
            intelligence_components: Intelligence components
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Trend analyst or None
        """
        try:
            # Extract pattern recognition intelligence
            pattern_intelligence = intelligence_components.get('pattern_recognition', {})
            
            # Create trend analyst based on regime characteristics
            analyst_params = regime_config.get('analyst_parameters', {}).get('trend_analyst', {})
            
            trend_analyst = {
                'analyst_type': 'trend_analyst',
                'regime_id': regime_id,
                'specialization': 'trend_analysis',
                'capabilities': {
                    'trend_detection': True,
                    'trend_strength_measurement': True,
                    'trend_persistence_analysis': True,
                    'trend_reversal_detection': True
                },
                'parameters': {
                    'trend_strength_threshold': analyst_params.get('trend_strength_threshold', 0.6),
                    'trend_persistence_required': analyst_params.get('trend_persistence_required', 3),
                    'trend_confirmation_methods': analyst_params.get('trend_confirmation_methods', ['price_action'])
                },
                'intelligence_integration': {
                    'pattern_rules': pattern_intelligence.get('pattern_rules', {}),
                    'confidence_thresholds': pattern_intelligence.get('confidence_thresholds', {}),
                    'pattern_types': pattern_intelligence.get('pattern_types', [])
                },
                'performance_metrics': {
                    'trend_accuracy': 0.0,  # Will be calculated during training
                    'trend_precision': 0.0,
                    'trend_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created trend analyst for regime {regime_id}")
            return trend_analyst
            
        except Exception as e:
            self.logger.error(f"❌ Error creating trend analyst for regime {regime_id}: {e}")
            return None
    
    async def _create_volatility_analyst(
        self,
        intelligence_components: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create volatility analyst for regime.
        
        Args:
            intelligence_components: Intelligence components
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Volatility analyst or None
        """
        try:
            # Extract risk assessment intelligence
            risk_intelligence = intelligence_components.get('risk_assessment', {})
            
            # Create volatility analyst based on regime characteristics
            analyst_params = regime_config.get('analyst_parameters', {}).get('volatility_analyst', {})
            
            volatility_analyst = {
                'analyst_type': 'volatility_analyst',
                'regime_id': regime_id,
                'specialization': 'volatility_analysis',
                'capabilities': {
                    'volatility_measurement': True,
                    'volatility_regime_detection': True,
                    'volatility_forecasting': True,
                    'volatility_risk_assessment': True
                },
                'parameters': {
                    'volatility_threshold': analyst_params.get('volatility_threshold', 0.8),
                    'volatility_analysis_methods': analyst_params.get('volatility_analysis_methods', ['ATR']),
                    'volatility_regime_detection': analyst_params.get('volatility_regime_detection', True)
                },
                'intelligence_integration': {
                    'risk_rules': risk_intelligence.get('risk_rules', {}),
                    'risk_tolerance': risk_intelligence.get('risk_tolerance', 'balanced'),
                    'regime_characteristics': risk_intelligence.get('regime_characteristics', {})
                },
                'performance_metrics': {
                    'volatility_accuracy': 0.0,
                    'volatility_precision': 0.0,
                    'volatility_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created volatility analyst for regime {regime_id}")
            return volatility_analyst
            
        except Exception as e:
            self.logger.error(f"❌ Error creating volatility analyst for regime {regime_id}: {e}")
            return None
    
    async def _create_momentum_analyst(
        self,
        intelligence_components: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create momentum analyst for regime.
        
        Args:
            intelligence_components: Intelligence components
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Momentum analyst or None
        """
        try:
            # Extract signal generation intelligence
            signal_intelligence = intelligence_components.get('signal_generation', {})
            
            # Create momentum analyst based on regime characteristics
            analyst_params = regime_config.get('analyst_parameters', {}).get('momentum_analyst', {})
            
            momentum_analyst = {
                'analyst_type': 'momentum_analyst',
                'regime_id': regime_id,
                'specialization': 'momentum_analysis',
                'capabilities': {
                    'momentum_detection': True,
                    'momentum_strength_measurement': True,
                    'momentum_divergence_analysis': True,
                    'momentum_continuation_prediction': True
                },
                'parameters': {
                    'momentum_threshold': analyst_params.get('momentum_threshold', 0.5),
                    'momentum_persistence_required': analyst_params.get('momentum_persistence_required', 2),
                    'momentum_indicators': analyst_params.get('momentum_indicators', ['RSI', 'MACD'])
                },
                'intelligence_integration': {
                    'signal_rules': signal_intelligence.get('signal_rules', {}),
                    'confidence_thresholds': signal_intelligence.get('confidence_thresholds', {}),
                    'signal_parameters': signal_intelligence.get('signal_parameters', {})
                },
                'performance_metrics': {
                    'momentum_accuracy': 0.0,
                    'momentum_precision': 0.0,
                    'momentum_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created momentum analyst for regime {regime_id}")
            return momentum_analyst
            
        except Exception as e:
            self.logger.error(f"❌ Error creating momentum analyst for regime {regime_id}: {e}")
            return None
    
    async def _create_volume_analyst(
        self,
        intelligence_components: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create volume analyst for regime.
        
        Args:
            intelligence_components: Intelligence components
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Volume analyst or None
        """
        try:
            # Create volume analyst based on regime characteristics
            analyst_params = regime_config.get('analyst_parameters', {}).get('volume_analyst', {})
            
            volume_analyst = {
                'analyst_type': 'volume_analyst',
                'regime_id': regime_id,
                'specialization': 'volume_analysis',
                'capabilities': {
                    'volume_analysis': True,
                    'volume_profile_analysis': True,
                    'volume_confirmation': True,
                    'volume_divergence_detection': True
                },
                'parameters': {
                    'volume_confirmation_required': analyst_params.get('volume_confirmation_required', True),
                    'volume_threshold': analyst_params.get('volume_threshold', 1.2),
                    'volume_analysis_methods': analyst_params.get('volume_analysis_methods', ['OBV'])
                },
                'intelligence_integration': {
                    'pattern_rules': intelligence_components.get('pattern_recognition', {}).get('pattern_rules', {}),
                    'signal_rules': intelligence_components.get('signal_generation', {}).get('signal_rules', {})
                },
                'performance_metrics': {
                    'volume_accuracy': 0.0,
                    'volume_precision': 0.0,
                    'volume_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created volume analyst for regime {regime_id}")
            return volume_analyst
            
        except Exception as e:
            self.logger.error(f"❌ Error creating volume analyst for regime {regime_id}: {e}")
            return None
    
    async def _create_risk_analyst(
        self,
        intelligence_components: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create risk analyst for regime.
        
        Args:
            intelligence_components: Intelligence components
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Risk analyst or None
        """
        try:
            # Extract risk assessment intelligence
            risk_intelligence = intelligence_components.get('risk_assessment', {})
            
            # Create risk analyst based on regime characteristics
            analyst_params = regime_config.get('analyst_parameters', {}).get('risk_analyst', {})
            
            risk_analyst = {
                'analyst_type': 'risk_analyst',
                'regime_id': regime_id,
                'specialization': 'risk_analysis',
                'capabilities': {
                    'risk_measurement': True,
                    'risk_monitoring': True,
                    'risk_control': True,
                    'risk_reporting': True
                },
                'parameters': {
                    'risk_tolerance': analyst_params.get('risk_tolerance', 'balanced'),
                    'max_drawdown_threshold': analyst_params.get('max_drawdown_threshold', 0.05),
                    'var_threshold': analyst_params.get('var_threshold', 0.02)
                },
                'intelligence_integration': {
                    'risk_rules': risk_intelligence.get('risk_rules', {}),
                    'risk_tolerance': risk_intelligence.get('risk_tolerance', 'balanced'),
                    'regime_characteristics': risk_intelligence.get('regime_characteristics', {})
                },
                'performance_metrics': {
                    'risk_accuracy': 0.0,
                    'risk_precision': 0.0,
                    'risk_recall': 0.0
                }
            }
            
            self.logger.info(f"✅ Created risk analyst for regime {regime_id}")
            return risk_analyst
            
        except Exception as e:
            self.logger.error(f"❌ Error creating risk analyst for regime {regime_id}: {e}")
            return None
    
    async def _create_ensemble_analyst(
        self,
        individual_analysts: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create ensemble analyst for regime.
        
        Args:
            individual_analysts: Individual analyst results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Ensemble analyst or None
        """
        try:
            if not individual_analysts:
                self.logger.warning(f"⚠️ No individual analysts available for ensemble creation in regime {regime_id}")
                return None
            
            # Create ensemble analyst based on individual analysts
            analyst_weights = regime_config.get('analyst_weights', {})
            
            ensemble_analyst = {
                'analyst_type': 'ensemble_analyst',
                'regime_id': regime_id,
                'specialization': 'ensemble_analysis',
                'capabilities': {
                    'ensemble_prediction': True,
                    'consensus_analysis': True,
                    'confidence_weighting': True,
                    'diversity_management': True
                },
                'parameters': {
                    'ensemble_method': 'weighted_voting',
                    'confidence_weighting': True,
                    'diversity_requirement': 0.3,
                    'consensus_threshold': 0.6
                },
                'individual_analysts': list(individual_analysts.keys()),
                'analyst_weights': analyst_weights,
                'ensemble_metadata': {
                    'analyst_count': len(individual_analysts),
                    'weight_distribution': analyst_weights,
                    'diversity_score': self._calculate_analyst_diversity(individual_analysts)
                },
                'performance_metrics': {
                    'ensemble_accuracy': 0.0,
                    'ensemble_precision': 0.0,
                    'ensemble_recall': 0.0,
                    'consensus_accuracy': 0.0
                }
            }
            
            self.logger.info(f"✅ Created ensemble analyst for regime {regime_id} with {len(individual_analysts)} individual analysts")
            return ensemble_analyst
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble analyst for regime {regime_id}: {e}")
            return None
    
    def _calculate_analyst_diversity(self, individual_analysts: Dict[str, Any]) -> float:
        """Calculate diversity score of individual analysts.
        
        Args:
            individual_analysts: Individual analyst results
            
        Returns:
            Diversity score
        """
        try:
            if len(individual_analysts) <= 1:
                return 0.0
            
            # Calculate diversity based on analyst specializations
            specializations = [analyst.get('specialization', 'unknown') for analyst in individual_analysts.values()]
            unique_specializations = len(set(specializations))
            
            # Normalize diversity score
            diversity_score = unique_specializations / len(specializations)
            
            return float(diversity_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating analyst diversity: {e}")
            return 0.0
    
    def _calculate_analyst_performance(self, created_analysts: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall analyst performance metrics.
        
        Args:
            created_analysts: Created analyst results
            
        Returns:
            Performance metrics
        """
        try:
            if not created_analysts:
                return {}
            
            # Calculate performance metrics for each analyst type
            performance_metrics = {}
            
            for analyst_name, analyst_data in created_analysts.items():
                if 'performance_metrics' in analyst_data:
                    performance_metrics[analyst_name] = analyst_data['performance_metrics']
            
            # Calculate overall performance
            overall_performance = {
                'total_analysts': len(created_analysts),
                'analyst_types': list(created_analysts.keys()),
                'performance_metrics': performance_metrics,
                'diversity_score': self._calculate_analyst_diversity(created_analysts),
                'specialization_coverage': len(set(analyst.get('specialization', 'unknown') for analyst in created_analysts.values()))
            }
            
            return overall_performance
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating analyst performance: {e}")
            return {}
    
    async def _save_regime_analyst_results(
        self,
        analyst_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save analyst creation results for a specific regime.
        
        Args:
            analyst_results: Analyst creation results
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
            analyst_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_analyst_creation_regime_{regime_id}.json'
            
            with open(analyst_path, 'w') as f:
                json.dump(analyst_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved analyst creation results for regime {regime_id}: {analyst_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving analyst creation results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_analyst_creation_step')
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
    """Run the enhanced per-regime analyst creation step.
    
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
    logger.info("🚀 Starting Step 11: Per-Regime Analyst Creation")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_analyst_creation'] = True
    
    # Initialize and run the per-regime analyst creation step
    step = PerRegimeAnalystCreationStep(config)
    
    success = await step.execute_per_regime_analyst_creation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 11: Per-Regime Analyst Creation completed successfully")
    else:
        logger.error("❌ Step 11: Per-Regime Analyst Creation failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime analyst creation step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime analyst creation result: {success}')
        
    asyncio.run(test())