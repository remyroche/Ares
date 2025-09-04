"""Step 10: Unified Regime Intelligence - Per-Regime Implementation.

This module provides per-HMM regime intelligence functionality, ensuring that
regime intelligence is developed specifically for each regime's characteristics.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step10_unified_regime_intelligence import Step10UnifiedRegimeIntelligence
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


logger = get_logger('Step10UnifiedRegimeIntelligencePerRegime')


class PerRegimeUnifiedRegimeIntelligenceStep(Step10UnifiedRegimeIntelligence):
    """Unified regime intelligence step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_regime_intelligence', True)
        self.regime_specific_configs = config.get('regime_specific_intelligence_configs', {})
        self.adaptive_intelligence_parameters = config.get('adaptive_intelligence_parameters_per_regime', True)
        
    @traced(span_name='execute_per_regime_regime_intelligence')
    @per_regime_step('step10_unified_regime_intelligence')
    async def execute_per_regime_regime_intelligence(
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
        """Execute unified regime intelligence on a per-regime basis.
        
        Each regime may have different intelligence requirements, so regime
        intelligence should be developed specifically for each regime's characteristics.
        
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
            self.logger.info(f"🚀 Starting per-regime regime intelligence for regime {regime_id}")
            
            # Load HMM training results from previous step
            training_data = await self._load_hmm_training_data(symbol, exchange, timeframe, data_dir, regime_id)
            if training_data is None:
                self.logger.error(f"❌ Failed to load HMM training data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_intelligence_config(regime_id)
            
            # Apply regime-specific intelligence development
            intelligence_results = await self._apply_regime_intelligence_development(
                training_data, regime_config, regime_id
            )
            
            if intelligence_results is None:
                self.logger.error(f"❌ Failed regime intelligence development for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_intelligence_results(
                intelligence_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed regime intelligence for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save intelligence results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime regime intelligence for regime {regime_id}: {e}")
            return False
    
    async def _load_hmm_training_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load HMM training data for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            HMM training data or None
        """
        try:
            # Try per-regime HMM training data first
            training_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_hmm_training_regime_{regime_id}.json'
            
            if not training_path.exists():
                # Fall back to aggregated HMM training data
                training_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_hmm_training_aggregated.json'
            
            if training_path.exists():
                with open(training_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"✅ Loaded HMM training data for regime {regime_id}")
                return data
            else:
                self.logger.error(f"❌ HMM training data not found: {training_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading HMM training data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_intelligence_config(self, regime_id: int) -> Dict[str, Any]:
        """Get regime intelligence configuration for a specific regime.
        
        Different regimes may require different intelligence strategies and parameters.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific intelligence configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_pattern_recognition': True,
            'enable_signal_generation': True,
            'enable_risk_assessment': True,
            'enable_performance_prediction': True,
            'enable_ensemble_intelligence': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following intelligence
            return {
                **base_config,
                'intelligence_strategy': {
                    'emphasis': 'trend_following',
                    'pattern_types': ['trend_continuation', 'momentum_breakout', 'trend_reversal'],
                    'signal_confidence_threshold': 0.7,
                    'risk_tolerance': 'moderate'
                },
                'intelligence_parameters': {
                    'pattern_recognition': {
                        'trend_strength_threshold': 0.6,
                        'momentum_threshold': 0.5,
                        'volume_confirmation_required': True
                    },
                    'signal_generation': {
                        'entry_confidence_min': 0.75,
                        'exit_confidence_min': 0.65,
                        'signal_persistence_required': 3
                    },
                    'risk_assessment': {
                        'max_position_size': 0.1,
                        'stop_loss_threshold': 0.02,
                        'take_profit_ratio': 2.0
                    }
                }
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize mean-reversion intelligence
            return {
                **base_config,
                'intelligence_strategy': {
                    'emphasis': 'mean_reversion',
                    'pattern_types': ['oversold_bounce', 'overbought_rejection', 'range_breakout'],
                    'signal_confidence_threshold': 0.8,
                    'risk_tolerance': 'conservative'
                },
                'intelligence_parameters': {
                    'pattern_recognition': {
                        'volatility_threshold': 0.8,
                        'mean_reversion_strength': 0.7,
                        'range_boundary_confirmation': True
                    },
                    'signal_generation': {
                        'entry_confidence_min': 0.85,
                        'exit_confidence_min': 0.75,
                        'signal_persistence_required': 2
                    },
                    'risk_assessment': {
                        'max_position_size': 0.05,
                        'stop_loss_threshold': 0.015,
                        'take_profit_ratio': 1.5
                    }
                }
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'intelligence_strategy': {
                    'emphasis': 'balanced',
                    'pattern_types': ['mixed_signals', 'adaptive_patterns', 'context_aware'],
                    'signal_confidence_threshold': 0.75,
                    'risk_tolerance': 'balanced'
                },
                'intelligence_parameters': {
                    'pattern_recognition': {
                        'adaptive_threshold': 0.65,
                        'context_weight': 0.3,
                        'multi_timeframe_confirmation': True
                    },
                    'signal_generation': {
                        'entry_confidence_min': 0.8,
                        'exit_confidence_min': 0.7,
                        'signal_persistence_required': 2
                    },
                    'risk_assessment': {
                        'max_position_size': 0.075,
                        'stop_loss_threshold': 0.0175,
                        'take_profit_ratio': 1.75
                    }
                }
            }
    
    async def _apply_regime_intelligence_development(
        self,
        training_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply regime intelligence development to training data.
        
        Args:
            training_data: HMM training results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Intelligence results or None
        """
        try:
            self.logger.info(f"🔧 Applying regime intelligence development for regime {regime_id}")
            
            # Extract model information
            models = training_data.get('models', {})
            if not models:
                self.logger.warning(f"⚠️ No models found for regime {regime_id}")
                return None
            
            results = {
                'regime_id': regime_id,
                'intelligence_strategy': regime_config.get('intelligence_strategy', {}),
                'intelligence_parameters': regime_config.get('intelligence_parameters', {}),
                'intelligence_components': {},
                'performance_metrics': {},
                'intelligence_metadata': {}
            }
            
            # Develop pattern recognition intelligence
            if regime_config.get('enable_pattern_recognition', True):
                pattern_intelligence = await self._develop_pattern_recognition_intelligence(
                    models, regime_config, regime_id
                )
                if pattern_intelligence:
                    results['intelligence_components']['pattern_recognition'] = pattern_intelligence
            
            # Develop signal generation intelligence
            if regime_config.get('enable_signal_generation', True):
                signal_intelligence = await self._develop_signal_generation_intelligence(
                    models, regime_config, regime_id
                )
                if signal_intelligence:
                    results['intelligence_components']['signal_generation'] = signal_intelligence
            
            # Develop risk assessment intelligence
            if regime_config.get('enable_risk_assessment', True):
                risk_intelligence = await self._develop_risk_assessment_intelligence(
                    models, regime_config, regime_id
                )
                if risk_intelligence:
                    results['intelligence_components']['risk_assessment'] = risk_intelligence
            
            # Develop performance prediction intelligence
            if regime_config.get('enable_performance_prediction', True):
                performance_intelligence = await self._develop_performance_prediction_intelligence(
                    models, regime_config, regime_id
                )
                if performance_intelligence:
                    results['intelligence_components']['performance_prediction'] = performance_intelligence
            
            # Develop ensemble intelligence
            if regime_config.get('enable_ensemble_intelligence', True):
                ensemble_intelligence = await self._develop_ensemble_intelligence(
                    models, regime_config, regime_id
                )
                if ensemble_intelligence:
                    results['intelligence_components']['ensemble_intelligence'] = ensemble_intelligence
            
            # Calculate overall intelligence metrics
            results['performance_metrics'] = self._calculate_intelligence_performance(results['intelligence_components'])
            
            self.logger.info(f"✅ Completed regime intelligence development for regime {regime_id}: {len(results['intelligence_components'])} components")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying regime intelligence development for regime {regime_id}: {e}")
            return None
    
    async def _develop_pattern_recognition_intelligence(
        self,
        models: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Develop pattern recognition intelligence for regime.
        
        Args:
            models: Model results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Pattern recognition intelligence or None
        """
        try:
            intelligence_params = regime_config.get('intelligence_parameters', {}).get('pattern_recognition', {})
            
            # Extract feature importance from models
            feature_importance = {}
            for model_name, model_data in models.items():
                if 'feature_importance' in model_data:
                    feature_importance[model_name] = model_data['feature_importance']
                elif 'feature_coefficients' in model_data:
                    feature_importance[model_name] = model_data['feature_coefficients']
            
            # Develop pattern recognition rules
            pattern_rules = self._develop_pattern_rules(feature_importance, intelligence_params, regime_id)
            
            # Create pattern recognition intelligence
            intelligence = {
                'intelligence_type': 'pattern_recognition',
                'regime_id': regime_id,
                'pattern_rules': pattern_rules,
                'feature_importance': feature_importance,
                'confidence_thresholds': {
                    'high_confidence': intelligence_params.get('trend_strength_threshold', 0.6),
                    'medium_confidence': intelligence_params.get('trend_strength_threshold', 0.6) * 0.8,
                    'low_confidence': intelligence_params.get('trend_strength_threshold', 0.6) * 0.6
                },
                'pattern_types': regime_config.get('intelligence_strategy', {}).get('pattern_types', [])
            }
            
            self.logger.info(f"✅ Developed pattern recognition intelligence for regime {regime_id}")
            return intelligence
            
        except Exception as e:
            self.logger.error(f"❌ Error developing pattern recognition intelligence for regime {regime_id}: {e}")
            return None
    
    def _develop_pattern_rules(
        self,
        feature_importance: Dict[str, List[float]],
        intelligence_params: Dict[str, Any],
        regime_id: int
    ) -> Dict[str, Any]:
        """Develop pattern recognition rules based on feature importance.
        
        Args:
            feature_importance: Feature importance from models
            intelligence_params: Intelligence parameters
            regime_id: Regime ID
            
        Returns:
            Pattern recognition rules
        """
        try:
            rules = {
                'trend_patterns': {},
                'momentum_patterns': {},
                'volatility_patterns': {},
                'volume_patterns': {}
            }
            
            # Analyze feature importance to develop rules
            for model_name, importance in feature_importance.items():
                if not importance:
                    continue
                
                # Find top features
                top_features = np.argsort(importance)[-5:]  # Top 5 features
                
                # Develop rules based on regime characteristics
                if regime_id <= 2:  # Trending regimes
                    rules['trend_patterns'][model_name] = {
                        'top_features': top_features.tolist(),
                        'trend_strength_threshold': intelligence_params.get('trend_strength_threshold', 0.6),
                        'momentum_confirmation': True
                    }
                elif regime_id >= 5:  # Volatile regimes
                    rules['volatility_patterns'][model_name] = {
                        'top_features': top_features.tolist(),
                        'volatility_threshold': intelligence_params.get('volatility_threshold', 0.8),
                        'mean_reversion_strength': intelligence_params.get('mean_reversion_strength', 0.7)
                    }
                else:  # Balanced regimes
                    rules['momentum_patterns'][model_name] = {
                        'top_features': top_features.tolist(),
                        'adaptive_threshold': intelligence_params.get('adaptive_threshold', 0.65),
                        'context_weight': intelligence_params.get('context_weight', 0.3)
                    }
            
            return rules
            
        except Exception as e:
            self.logger.error(f"❌ Error developing pattern rules for regime {regime_id}: {e}")
            return {}
    
    async def _develop_signal_generation_intelligence(
        self,
        models: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Develop signal generation intelligence for regime.
        
        Args:
            models: Model results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Signal generation intelligence or None
        """
        try:
            intelligence_params = regime_config.get('intelligence_parameters', {}).get('signal_generation', {})
            
            # Analyze model predictions to develop signal generation rules
            signal_rules = {}
            
            for model_name, model_data in models.items():
                if 'probabilities' in model_data:
                    probabilities = model_data['probabilities']
                    
                    # Calculate signal generation parameters
                    signal_rules[model_name] = {
                        'entry_threshold': intelligence_params.get('entry_confidence_min', 0.8),
                        'exit_threshold': intelligence_params.get('exit_confidence_min', 0.7),
                        'signal_persistence': intelligence_params.get('signal_persistence_required', 2),
                        'probability_distribution': {
                            'mean': float(np.mean(probabilities)),
                            'std': float(np.std(probabilities)),
                            'min': float(np.min(probabilities)),
                            'max': float(np.max(probabilities))
                        }
                    }
            
            intelligence = {
                'intelligence_type': 'signal_generation',
                'regime_id': regime_id,
                'signal_rules': signal_rules,
                'confidence_thresholds': {
                    'entry_confidence': intelligence_params.get('entry_confidence_min', 0.8),
                    'exit_confidence': intelligence_params.get('exit_confidence_min', 0.7)
                },
                'signal_parameters': {
                    'persistence_required': intelligence_params.get('signal_persistence_required', 2),
                    'confirmation_models': len(signal_rules)
                }
            }
            
            self.logger.info(f"✅ Developed signal generation intelligence for regime {regime_id}")
            return intelligence
            
        except Exception as e:
            self.logger.error(f"❌ Error developing signal generation intelligence for regime {regime_id}: {e}")
            return None
    
    async def _develop_risk_assessment_intelligence(
        self,
        models: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Develop risk assessment intelligence for regime.
        
        Args:
            models: Model results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Risk assessment intelligence or None
        """
        try:
            intelligence_params = regime_config.get('intelligence_parameters', {}).get('risk_assessment', {})
            
            # Develop risk assessment rules based on regime characteristics
            risk_rules = {
                'position_sizing': {
                    'max_position_size': intelligence_params.get('max_position_size', 0.1),
                    'regime_risk_multiplier': self._calculate_regime_risk_multiplier(regime_id)
                },
                'stop_loss': {
                    'threshold': intelligence_params.get('stop_loss_threshold', 0.02),
                    'dynamic_adjustment': True
                },
                'take_profit': {
                    'ratio': intelligence_params.get('take_profit_ratio', 2.0),
                    'scaling_enabled': True
                },
                'risk_metrics': {
                    'var_95': self._calculate_var_95(models),
                    'expected_shortfall': self._calculate_expected_shortfall(models),
                    'sharpe_ratio': self._calculate_sharpe_ratio(models)
                }
            }
            
            intelligence = {
                'intelligence_type': 'risk_assessment',
                'regime_id': regime_id,
                'risk_rules': risk_rules,
                'risk_tolerance': regime_config.get('intelligence_strategy', {}).get('risk_tolerance', 'balanced'),
                'regime_characteristics': {
                    'volatility_level': 'high' if regime_id >= 5 else 'low' if regime_id <= 2 else 'medium',
                    'trend_strength': 'high' if regime_id <= 2 else 'low' if regime_id >= 5 else 'medium'
                }
            }
            
            self.logger.info(f"✅ Developed risk assessment intelligence for regime {regime_id}")
            return intelligence
            
        except Exception as e:
            self.logger.error(f"❌ Error developing risk assessment intelligence for regime {regime_id}: {e}")
            return None
    
    def _calculate_regime_risk_multiplier(self, regime_id: int) -> float:
        """Calculate risk multiplier based on regime characteristics.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Risk multiplier
        """
        if regime_id <= 2:  # Trending regimes - moderate risk
            return 1.0
        elif regime_id >= 5:  # Volatile regimes - high risk
            return 0.5
        else:  # Balanced regimes - balanced risk
            return 0.75
    
    def _calculate_var_95(self, models: Dict[str, Any]) -> float:
        """Calculate Value at Risk (95%) from model predictions.
        
        Args:
            models: Model results
            
        Returns:
            VaR 95% value
        """
        try:
            all_probabilities = []
            for model_data in models.values():
                if 'probabilities' in model_data:
                    all_probabilities.extend(model_data['probabilities'])
            
            if all_probabilities:
                return float(np.percentile(all_probabilities, 5))  # 5th percentile
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating VaR 95%: {e}")
            return 0.0
    
    def _calculate_expected_shortfall(self, models: Dict[str, Any]) -> float:
        """Calculate Expected Shortfall from model predictions.
        
        Args:
            models: Model results
            
        Returns:
            Expected Shortfall value
        """
        try:
            all_probabilities = []
            for model_data in models.values():
                if 'probabilities' in model_data:
                    all_probabilities.extend(model_data['probabilities'])
            
            if all_probabilities:
                var_95 = np.percentile(all_probabilities, 5)
                tail_losses = [p for p in all_probabilities if p <= var_95]
                return float(np.mean(tail_losses)) if tail_losses else 0.0
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Expected Shortfall: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, models: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio from model predictions.
        
        Args:
            models: Model results
            
        Returns:
            Sharpe ratio
        """
        try:
            all_probabilities = []
            for model_data in models.values():
                if 'probabilities' in model_data:
                    all_probabilities.extend(model_data['probabilities'])
            
            if all_probabilities and len(all_probabilities) > 1:
                returns = np.diff(all_probabilities)
                if np.std(returns) > 0:
                    return float(np.mean(returns) / np.std(returns))
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _develop_performance_prediction_intelligence(
        self,
        models: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Develop performance prediction intelligence for regime.
        
        Args:
            models: Model results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Performance prediction intelligence or None
        """
        try:
            # Analyze model performance to develop prediction intelligence
            performance_metrics = {}
            
            for model_name, model_data in models.items():
                if 'accuracy' in model_data:
                    performance_metrics[model_name] = {
                        'accuracy': model_data['accuracy'],
                        'confidence': model_data.get('accuracy', 0.0),
                        'reliability': self._calculate_model_reliability(model_data)
                    }
            
            # Develop performance prediction rules
            prediction_rules = {
                'performance_thresholds': {
                    'excellent': 0.8,
                    'good': 0.7,
                    'fair': 0.6,
                    'poor': 0.5
                },
                'regime_performance_expectations': {
                    'trending_regimes': {'min_accuracy': 0.65, 'target_accuracy': 0.75},
                    'volatile_regimes': {'min_accuracy': 0.60, 'target_accuracy': 0.70},
                    'balanced_regimes': {'min_accuracy': 0.62, 'target_accuracy': 0.72}
                }
            }
            
            intelligence = {
                'intelligence_type': 'performance_prediction',
                'regime_id': regime_id,
                'performance_metrics': performance_metrics,
                'prediction_rules': prediction_rules,
                'expected_performance': self._calculate_expected_performance(regime_id, performance_metrics)
            }
            
            self.logger.info(f"✅ Developed performance prediction intelligence for regime {regime_id}")
            return intelligence
            
        except Exception as e:
            self.logger.error(f"❌ Error developing performance prediction intelligence for regime {regime_id}: {e}")
            return None
    
    def _calculate_model_reliability(self, model_data: Dict[str, Any]) -> float:
        """Calculate model reliability score.
        
        Args:
            model_data: Model data
            
        Returns:
            Reliability score
        """
        try:
            accuracy = model_data.get('accuracy', 0.0)
            # Simple reliability calculation based on accuracy
            return min(1.0, accuracy * 1.2)  # Boost slightly for reliability
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating model reliability: {e}")
            return 0.0
    
    def _calculate_expected_performance(self, regime_id: int, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected performance for regime.
        
        Args:
            regime_id: Regime ID
            performance_metrics: Performance metrics
            
        Returns:
            Expected performance
        """
        try:
            if not performance_metrics:
                return {'expected_accuracy': 0.5, 'confidence': 0.0}
            
            # Calculate weighted average performance
            total_weight = 0
            weighted_accuracy = 0
            
            for model_name, metrics in performance_metrics.items():
                weight = metrics.get('reliability', 0.5)
                accuracy = metrics.get('accuracy', 0.0)
                
                weighted_accuracy += accuracy * weight
                total_weight += weight
            
            expected_accuracy = weighted_accuracy / total_weight if total_weight > 0 else 0.5
            
            return {
                'expected_accuracy': float(expected_accuracy),
                'confidence': float(min(1.0, total_weight / len(performance_metrics))),
                'model_count': len(performance_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating expected performance: {e}")
            return {'expected_accuracy': 0.5, 'confidence': 0.0}
    
    async def _develop_ensemble_intelligence(
        self,
        models: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Develop ensemble intelligence for regime.
        
        Args:
            models: Model results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Ensemble intelligence or None
        """
        try:
            # Analyze ensemble performance
            ensemble_data = models.get('ensemble', {})
            if not ensemble_data:
                # Create ensemble from individual models
                ensemble_data = self._create_ensemble_from_individual_models(models)
            
            # Develop ensemble intelligence
            intelligence = {
                'intelligence_type': 'ensemble_intelligence',
                'regime_id': regime_id,
                'ensemble_performance': ensemble_data.get('accuracy', 0.0),
                'individual_models': list(models.keys()),
                'ensemble_strategy': {
                    'voting_method': 'weighted_average',
                    'confidence_weighting': True,
                    'model_diversity': len(models)
                },
                'ensemble_parameters': {
                    'min_models_required': 2,
                    'confidence_threshold': regime_config.get('intelligence_strategy', {}).get('signal_confidence_threshold', 0.75),
                    'weight_calculation': 'performance_based'
                }
            }
            
            self.logger.info(f"✅ Developed ensemble intelligence for regime {regime_id}")
            return intelligence
            
        except Exception as e:
            self.logger.error(f"❌ Error developing ensemble intelligence for regime {regime_id}: {e}")
            return None
    
    def _create_ensemble_from_individual_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble data from individual models.
        
        Args:
            models: Individual model results
            
        Returns:
            Ensemble data
        """
        try:
            if not models:
                return {}
            
            # Calculate ensemble accuracy as average of individual models
            accuracies = [model_data.get('accuracy', 0.0) for model_data in models.values()]
            ensemble_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            return {
                'accuracy': float(ensemble_accuracy),
                'model_count': len(models),
                'individual_accuracies': accuracies
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble from individual models: {e}")
            return {}
    
    def _calculate_intelligence_performance(self, intelligence_components: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall intelligence performance metrics.
        
        Args:
            intelligence_components: Intelligence components
            
        Returns:
            Performance metrics
        """
        try:
            if not intelligence_components:
                return {}
            
            # Calculate component scores
            component_scores = {}
            for component_name, component_data in intelligence_components.items():
                if component_name == 'pattern_recognition':
                    # Score based on pattern rules complexity
                    pattern_rules = component_data.get('pattern_rules', {})
                    component_scores[component_name] = len(pattern_rules) / 10.0  # Normalize
                elif component_name == 'signal_generation':
                    # Score based on signal rules
                    signal_rules = component_data.get('signal_rules', {})
                    component_scores[component_name] = len(signal_rules) / 5.0  # Normalize
                elif component_name == 'risk_assessment':
                    # Score based on risk rules completeness
                    risk_rules = component_data.get('risk_rules', {})
                    component_scores[component_name] = len(risk_rules) / 4.0  # Normalize
                elif component_name == 'performance_prediction':
                    # Score based on expected performance
                    expected_perf = component_data.get('expected_performance', {})
                    component_scores[component_name] = expected_perf.get('expected_accuracy', 0.5)
                elif component_name == 'ensemble_intelligence':
                    # Score based on ensemble performance
                    ensemble_perf = component_data.get('ensemble_performance', 0.0)
                    component_scores[component_name] = ensemble_perf
            
            # Calculate overall intelligence score
            overall_score = np.mean(list(component_scores.values())) if component_scores else 0.0
            
            return {
                'overall_intelligence_score': float(overall_score),
                'component_scores': component_scores,
                'component_count': len(intelligence_components),
                'intelligence_completeness': len(intelligence_components) / 5.0  # 5 expected components
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating intelligence performance: {e}")
            return {}
    
    async def _save_regime_intelligence_results(
        self,
        intelligence_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save regime intelligence results for a specific regime.
        
        Args:
            intelligence_results: Intelligence results
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
            intelligence_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_intelligence_regime_{regime_id}.json'
            
            with open(intelligence_path, 'w') as f:
                json.dump(intelligence_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved regime intelligence results for regime {regime_id}: {intelligence_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving regime intelligence results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_regime_intelligence_step')
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
    """Run the enhanced per-regime unified regime intelligence step.
    
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
    logger.info("🚀 Starting Step 10: Per-Regime Unified Regime Intelligence")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_regime_intelligence'] = True
    
    # Initialize and run the per-regime regime intelligence step
    step = PerRegimeUnifiedRegimeIntelligenceStep(config)
    
    success = await step.execute_per_regime_regime_intelligence(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 10: Per-Regime Unified Regime Intelligence completed successfully")
    else:
        logger.error("❌ Step 10: Per-Regime Unified Regime Intelligence failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime regime intelligence step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime regime intelligence result: {success}')
        
    asyncio.run(test())