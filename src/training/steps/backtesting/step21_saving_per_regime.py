"""Step 21: Saving - Per-Regime Implementation.

This module provides per-HMM regime saving functionality, ensuring that
all regime-specific results are properly saved and aggregated for final use.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime

from src.training.steps.step21_saving import Step21Saving
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


logger = get_logger('Step21SavingPerRegime')


class PerRegimeSavingStep(Step21Saving):
    """Saving step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_saving', True)
        self.regime_specific_configs = config.get('regime_specific_saving_configs', {})
        self.adaptive_saving_strategies = config.get('adaptive_saving_strategies_per_regime', True)
        
    @traced(span_name='execute_per_regime_saving')
    @per_regime_step('step21_saving')
    async def execute_per_regime_saving(
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
        """Execute saving on a per-regime basis.
        
        Each regime may require different saving strategies, so results
        should be saved specifically for each regime's characteristics.
        
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
            self.logger.info(f"🚀 Starting per-regime saving for regime {regime_id}")
            
            # Load all previous step results for this regime
            all_regime_results = await self._load_all_regime_results(symbol, exchange, timeframe, data_dir, regime_id)
            if all_regime_results is None:
                self.logger.error(f"❌ Failed to load regime results for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_saving_config(regime_id)
            
            # Apply regime-specific saving
            saving_results = await self._apply_regime_saving(
                all_regime_results, regime_config, regime_id
            )
            
            if saving_results is None:
                self.logger.error(f"❌ Failed saving for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_saving_results(
                saving_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed saving for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime saving for regime {regime_id}: {e}")
            return False
    
    async def _load_all_regime_results(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load all results from previous steps for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            All regime results or None
        """
        try:
            training_dir = Path(data_dir) / 'training'
            all_results = {}
            
            # Define all possible step result files
            step_files = [
                'feature_engineering',
                'matrix_operations', 
                'feature_selection',
                'hmm_training',
                'regime_intelligence',
                'analyst_creation',
                'parameters_optimization'
            ]
            
            # Load results from each step
            for step_name in step_files:
                result_file = training_dir / f'{exchange}_{symbol}_{timeframe}_{step_name}_regime_{regime_id}.json'
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            all_results[step_name] = json.load(f)
                        self.logger.debug(f"✅ Loaded {step_name} results for regime {regime_id}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error loading {step_name} results for regime {regime_id}: {e}")
                else:
                    self.logger.debug(f"ℹ️ No {step_name} results found for regime {regime_id}")
            
            if all_results:
                self.logger.info(f"✅ Loaded {len(all_results)} step results for regime {regime_id}: {list(all_results.keys())}")
                return all_results
            else:
                self.logger.error(f"❌ No step results found for regime {regime_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading all regime results for regime {regime_id}: {e}")
            return None
    
    def _get_regime_saving_config(self, regime_id: int) -> Dict[str, Any]:
        """Get saving configuration for a specific regime.
        
        Different regimes may require different saving strategies and formats.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific saving configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'save_models': True,
            'save_parameters': True,
            'save_metadata': True,
            'save_performance_metrics': True,
            'save_analyst_data': True,
            'save_intelligence_data': True,
            'save_optimization_results': True,
            'create_regime_summary': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-related data saving
            return {
                **base_config,
                'saving_strategy': {
                    'emphasis': 'trend_data',
                    'compression_level': 'medium',
                    'include_visualizations': True
                },
                'saving_parameters': {
                    'model_format': 'pickle',
                    'metadata_format': 'json',
                    'performance_format': 'json',
                    'include_feature_importance': True,
                    'include_trend_analysis': True
                },
                'file_organization': {
                    'create_regime_subfolder': True,
                    'separate_model_files': True,
                    'include_timestamp': True
                }
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize volatility and risk data saving
            return {
                **base_config,
                'saving_strategy': {
                    'emphasis': 'volatility_data',
                    'compression_level': 'high',
                    'include_visualizations': True
                },
                'saving_parameters': {
                    'model_format': 'pickle',
                    'metadata_format': 'json',
                    'performance_format': 'json',
                    'include_risk_metrics': True,
                    'include_volatility_analysis': True
                },
                'file_organization': {
                    'create_regime_subfolder': True,
                    'separate_model_files': True,
                    'include_timestamp': True
                }
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'saving_strategy': {
                    'emphasis': 'balanced_data',
                    'compression_level': 'medium',
                    'include_visualizations': True
                },
                'saving_parameters': {
                    'model_format': 'pickle',
                    'metadata_format': 'json',
                    'performance_format': 'json',
                    'include_balanced_analysis': True,
                    'include_ensemble_data': True
                },
                'file_organization': {
                    'create_regime_subfolder': True,
                    'separate_model_files': True,
                    'include_timestamp': True
                }
            }
    
    async def _apply_regime_saving(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply saving to regime results.
        
        Args:
            all_regime_results: All results from previous steps
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Saving results or None
        """
        try:
            self.logger.info(f"🔧 Applying saving for regime {regime_id}")
            
            results = {
                'regime_id': regime_id,
                'saving_strategy': regime_config.get('saving_strategy', {}),
                'saving_parameters': regime_config.get('saving_parameters', {}),
                'saved_components': {},
                'saving_metadata': {},
                'file_paths': {}
            }
            
            # Save models
            if regime_config.get('save_models', True):
                model_results = await self._save_regime_models(
                    all_regime_results, regime_config, regime_id
                )
                if model_results:
                    results['saved_components']['models'] = model_results
            
            # Save parameters
            if regime_config.get('save_parameters', True):
                parameter_results = await self._save_regime_parameters(
                    all_regime_results, regime_config, regime_id
                )
                if parameter_results:
                    results['saved_components']['parameters'] = parameter_results
            
            # Save metadata
            if regime_config.get('save_metadata', True):
                metadata_results = await self._save_regime_metadata(
                    all_regime_results, regime_config, regime_id
                )
                if metadata_results:
                    results['saved_components']['metadata'] = metadata_results
            
            # Save performance metrics
            if regime_config.get('save_performance_metrics', True):
                performance_results = await self._save_regime_performance_metrics(
                    all_regime_results, regime_config, regime_id
                )
                if performance_results:
                    results['saved_components']['performance_metrics'] = performance_results
            
            # Save analyst data
            if regime_config.get('save_analyst_data', True):
                analyst_results = await self._save_regime_analyst_data(
                    all_regime_results, regime_config, regime_id
                )
                if analyst_results:
                    results['saved_components']['analyst_data'] = analyst_results
            
            # Save intelligence data
            if regime_config.get('save_intelligence_data', True):
                intelligence_results = await self._save_regime_intelligence_data(
                    all_regime_results, regime_config, regime_id
                )
                if intelligence_results:
                    results['saved_components']['intelligence_data'] = intelligence_results
            
            # Save optimization results
            if regime_config.get('save_optimization_results', True):
                optimization_results = await self._save_regime_optimization_results(
                    all_regime_results, regime_config, regime_id
                )
                if optimization_results:
                    results['saved_components']['optimization_results'] = optimization_results
            
            # Create regime summary
            if regime_config.get('create_regime_summary', True):
                summary_results = await self._create_regime_summary(
                    all_regime_results, regime_config, regime_id
                )
                if summary_results:
                    results['saved_components']['regime_summary'] = summary_results
            
            # Calculate saving metadata
            results['saving_metadata'] = self._calculate_saving_metadata(results['saved_components'])
            
            self.logger.info(f"✅ Completed saving for regime {regime_id}: {len(results['saved_components'])} components saved")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying saving for regime {regime_id}: {e}")
            return None
    
    async def _save_regime_models(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save models for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Model saving results or None
        """
        try:
            # Extract model data from HMM training results
            hmm_training = all_regime_results.get('hmm_training', {})
            models = hmm_training.get('models', {})
            
            if not models:
                self.logger.warning(f"⚠️ No models found for regime {regime_id}")
                return None
            
            # Create model saving results
            model_results = {
                'saved_models': list(models.keys()),
                'model_count': len(models),
                'model_types': [model_data.get('model_type', 'unknown') for model_data in models.values()],
                'model_performances': {name: model_data.get('accuracy', 0.0) for name, model_data in models.items()},
                'saving_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Saved {len(models)} models for regime {regime_id}")
            return model_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving models for regime {regime_id}: {e}")
            return None
    
    async def _save_regime_parameters(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save parameters for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Parameter saving results or None
        """
        try:
            # Extract parameter data from optimization results
            optimization = all_regime_results.get('parameters_optimization', {})
            optimization_results = optimization.get('optimization_results', {})
            
            if not optimization_results:
                self.logger.warning(f"⚠️ No optimization results found for regime {regime_id}")
                return None
            
            # Create parameter saving results
            parameter_results = {
                'optimized_parameters': {},
                'parameter_count': 0,
                'optimization_types': list(optimization_results.keys()),
                'saving_timestamp': datetime.now().isoformat()
            }
            
            # Extract optimized parameters
            for opt_type, opt_data in optimization_results.items():
                if 'optimized_parameters' in opt_data:
                    parameter_results['optimized_parameters'][opt_type] = opt_data['optimized_parameters']
                    parameter_results['parameter_count'] += len(opt_data['optimized_parameters'])
            
            self.logger.info(f"✅ Saved {parameter_results['parameter_count']} parameters for regime {regime_id}")
            return parameter_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving parameters for regime {regime_id}: {e}")
            return None
    
    async def _save_regime_metadata(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save metadata for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Metadata saving results or None
        """
        try:
            # Create comprehensive metadata
            metadata_results = {
                'regime_id': regime_id,
                'regime_characteristics': self._extract_regime_characteristics(regime_id),
                'processing_timeline': self._extract_processing_timeline(all_regime_results),
                'data_statistics': self._extract_data_statistics(all_regime_results),
                'component_versions': self._extract_component_versions(),
                'saving_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Saved metadata for regime {regime_id}")
            return metadata_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving metadata for regime {regime_id}: {e}")
            return None
    
    def _extract_regime_characteristics(self, regime_id: int) -> Dict[str, Any]:
        """Extract regime characteristics.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Regime characteristics
        """
        if regime_id <= 2:
            return {
                'market_type': 'trending',
                'volatility_level': 'low',
                'trend_strength': 'high',
                'mean_reversion': 'low'
            }
        elif regime_id >= 5:
            return {
                'market_type': 'volatile',
                'volatility_level': 'high',
                'trend_strength': 'low',
                'mean_reversion': 'high'
            }
        else:
            return {
                'market_type': 'balanced',
                'volatility_level': 'medium',
                'trend_strength': 'medium',
                'mean_reversion': 'medium'
            }
    
    def _extract_processing_timeline(self, all_regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract processing timeline from results.
        
        Args:
            all_regime_results: All regime results
            
        Returns:
            Processing timeline
        """
        timeline = {}
        
        for step_name, step_results in all_regime_results.items():
            if isinstance(step_results, dict):
                # Extract any timestamp information
                if 'timestamp' in step_results:
                    timeline[step_name] = step_results['timestamp']
                elif 'created_at' in step_results:
                    timeline[step_name] = step_results['created_at']
                else:
                    timeline[step_name] = 'unknown'
        
        return timeline
    
    def _extract_data_statistics(self, all_regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data statistics from results.
        
        Args:
            all_regime_results: All regime results
            
        Returns:
            Data statistics
        """
        statistics = {
            'total_steps': len(all_regime_results),
            'step_names': list(all_regime_results.keys()),
            'data_sizes': {}
        }
        
        for step_name, step_results in all_regime_results.items():
            if isinstance(step_results, dict):
                # Estimate data size
                data_size = len(str(step_results))
                statistics['data_sizes'][step_name] = data_size
        
        return statistics
    
    def _extract_component_versions(self) -> Dict[str, str]:
        """Extract component versions.
        
        Returns:
            Component versions
        """
        return {
            'per_regime_saving': '1.0.0',
            'regime_continuity_manager': '1.0.0',
            'pipeline_standards': '1.0.0',
            'saving_timestamp': datetime.now().isoformat()
        }
    
    async def _save_regime_performance_metrics(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save performance metrics for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Performance metrics saving results or None
        """
        try:
            # Extract performance metrics from all steps
            performance_metrics = {}
            
            for step_name, step_results in all_regime_results.items():
                if isinstance(step_results, dict):
                    # Look for performance metrics
                    if 'performance_metrics' in step_results:
                        performance_metrics[step_name] = step_results['performance_metrics']
                    elif 'performance' in step_results:
                        performance_metrics[step_name] = step_results['performance']
            
            if not performance_metrics:
                self.logger.warning(f"⚠️ No performance metrics found for regime {regime_id}")
                return None
            
            # Create performance metrics saving results
            performance_results = {
                'performance_metrics': performance_metrics,
                'step_count': len(performance_metrics),
                'overall_performance': self._calculate_overall_performance(performance_metrics),
                'saving_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Saved performance metrics for regime {regime_id}")
            return performance_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving performance metrics for regime {regime_id}: {e}")
            return None
    
    def _calculate_overall_performance(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance from step metrics.
        
        Args:
            performance_metrics: Performance metrics from all steps
            
        Returns:
            Overall performance metrics
        """
        try:
            overall_performance = {
                'total_steps': len(performance_metrics),
                'step_performances': {},
                'average_performance': 0.0
            }
            
            all_scores = []
            
            for step_name, metrics in performance_metrics.items():
                if isinstance(metrics, dict):
                    # Extract various performance scores
                    scores = []
                    
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and 0 <= value <= 1:
                            scores.append(value)
                    
                    if scores:
                        step_avg = np.mean(scores)
                        overall_performance['step_performances'][step_name] = step_avg
                        all_scores.append(step_avg)
            
            if all_scores:
                overall_performance['average_performance'] = float(np.mean(all_scores))
            
            return overall_performance
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating overall performance: {e}")
            return {'average_performance': 0.0}
    
    async def _save_regime_analyst_data(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save analyst data for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Analyst data saving results or None
        """
        try:
            # Extract analyst data
            analyst_creation = all_regime_results.get('analyst_creation', {})
            created_analysts = analyst_creation.get('created_analysts', {})
            
            if not created_analysts:
                self.logger.warning(f"⚠️ No analyst data found for regime {regime_id}")
                return None
            
            # Create analyst data saving results
            analyst_results = {
                'created_analysts': list(created_analysts.keys()),
                'analyst_count': len(created_analysts),
                'analyst_types': [analyst.get('analyst_type', 'unknown') for analyst in created_analysts.values()],
                'analyst_specializations': [analyst.get('specialization', 'unknown') for analyst in created_analysts.values()],
                'saving_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Saved analyst data for regime {regime_id}")
            return analyst_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving analyst data for regime {regime_id}: {e}")
            return None
    
    async def _save_regime_intelligence_data(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save intelligence data for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Intelligence data saving results or None
        """
        try:
            # Extract intelligence data
            regime_intelligence = all_regime_results.get('regime_intelligence', {})
            intelligence_components = regime_intelligence.get('intelligence_components', {})
            
            if not intelligence_components:
                self.logger.warning(f"⚠️ No intelligence data found for regime {regime_id}")
                return None
            
            # Create intelligence data saving results
            intelligence_results = {
                'intelligence_components': list(intelligence_components.keys()),
                'component_count': len(intelligence_components),
                'intelligence_strategy': regime_intelligence.get('intelligence_strategy', {}),
                'performance_metrics': regime_intelligence.get('performance_metrics', {}),
                'saving_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Saved intelligence data for regime {regime_id}")
            return intelligence_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving intelligence data for regime {regime_id}: {e}")
            return None
    
    async def _save_regime_optimization_results(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Save optimization results for regime.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Optimization results saving results or None
        """
        try:
            # Extract optimization results
            parameters_optimization = all_regime_results.get('parameters_optimization', {})
            optimization_results = parameters_optimization.get('optimization_results', {})
            
            if not optimization_results:
                self.logger.warning(f"⚠️ No optimization results found for regime {regime_id}")
                return None
            
            # Create optimization results saving results
            optimization_saving_results = {
                'optimization_types': list(optimization_results.keys()),
                'optimization_count': len(optimization_results),
                'optimization_strategy': parameters_optimization.get('optimization_strategy', {}),
                'performance_metrics': parameters_optimization.get('performance_metrics', {}),
                'saving_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Saved optimization results for regime {regime_id}")
            return optimization_saving_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saving optimization results for regime {regime_id}: {e}")
            return None
    
    async def _create_regime_summary(
        self,
        all_regime_results: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create regime summary.
        
        Args:
            all_regime_results: All regime results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Regime summary or None
        """
        try:
            # Create comprehensive regime summary
            regime_summary = {
                'regime_id': regime_id,
                'regime_characteristics': self._extract_regime_characteristics(regime_id),
                'processing_summary': {
                    'total_steps_processed': len(all_regime_results),
                    'steps_completed': list(all_regime_results.keys()),
                    'processing_status': 'completed'
                },
                'performance_summary': self._create_performance_summary(all_regime_results),
                'component_summary': self._create_component_summary(all_regime_results),
                'recommendations': self._generate_regime_recommendations(regime_id, all_regime_results),
                'created_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Created regime summary for regime {regime_id}")
            return regime_summary
            
        except Exception as e:
            self.logger.error(f"❌ Error creating regime summary for regime {regime_id}: {e}")
            return None
    
    def _create_performance_summary(self, all_regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance summary from all results.
        
        Args:
            all_regime_results: All regime results
            
        Returns:
            Performance summary
        """
        try:
            performance_summary = {
                'overall_performance': 0.0,
                'step_performances': {},
                'best_performing_step': None,
                'worst_performing_step': None
            }
            
            step_scores = {}
            
            for step_name, step_results in all_regime_results.items():
                if isinstance(step_results, dict):
                    # Extract performance scores
                    scores = []
                    
                    # Look for various performance indicators
                    for key in ['accuracy', 'performance', 'score', 'f1_score']:
                        if key in step_results:
                            value = step_results[key]
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                scores.append(value)
                    
                    # Look in nested performance_metrics
                    if 'performance_metrics' in step_results:
                        perf_metrics = step_results['performance_metrics']
                        if isinstance(perf_metrics, dict):
                            for key, value in perf_metrics.items():
                                if isinstance(value, (int, float)) and 0 <= value <= 1:
                                    scores.append(value)
                    
                    if scores:
                        avg_score = np.mean(scores)
                        step_scores[step_name] = avg_score
                        performance_summary['step_performances'][step_name] = avg_score
            
            if step_scores:
                performance_summary['overall_performance'] = float(np.mean(list(step_scores.values())))
                performance_summary['best_performing_step'] = max(step_scores.keys(), key=lambda k: step_scores[k])
                performance_summary['worst_performing_step'] = min(step_scores.keys(), key=lambda k: step_scores[k])
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"❌ Error creating performance summary: {e}")
            return {'overall_performance': 0.0}
    
    def _create_component_summary(self, all_regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create component summary from all results.
        
        Args:
            all_regime_results: All regime results
            
        Returns:
            Component summary
        """
        try:
            component_summary = {
                'total_components': len(all_regime_results),
                'component_types': {},
                'data_sizes': {}
            }
            
            for step_name, step_results in all_regime_results.items():
                # Categorize component types
                if 'models' in step_results:
                    component_summary['component_types'][step_name] = 'model_based'
                elif 'analysts' in step_results:
                    component_summary['component_types'][step_name] = 'analyst_based'
                elif 'intelligence' in step_results:
                    component_summary['component_types'][step_name] = 'intelligence_based'
                elif 'optimization' in step_results:
                    component_summary['component_types'][step_name] = 'optimization_based'
                else:
                    component_summary['component_types'][step_name] = 'data_based'
                
                # Estimate data size
                component_summary['data_sizes'][step_name] = len(str(step_results))
            
            return component_summary
            
        except Exception as e:
            self.logger.error(f"❌ Error creating component summary: {e}")
            return {'total_components': 0}
    
    def _generate_regime_recommendations(self, regime_id: int, all_regime_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for regime.
        
        Args:
            regime_id: Regime ID
            all_regime_results: All regime results
            
        Returns:
            List of recommendations
        """
        try:
            recommendations = []
            
            # Regime-specific recommendations
            if regime_id <= 2:
                recommendations.extend([
                    "This is a trending regime - focus on trend-following strategies",
                    "Consider longer lookback periods for trend analysis",
                    "Monitor for trend reversals and adjust accordingly"
                ])
            elif regime_id >= 5:
                recommendations.extend([
                    "This is a volatile regime - use conservative position sizing",
                    "Focus on mean-reversion strategies",
                    "Implement strict risk management controls"
                ])
            else:
                recommendations.extend([
                    "This is a balanced regime - use adaptive strategies",
                    "Monitor regime transitions closely",
                    "Balance trend-following and mean-reversion approaches"
                ])
            
            # Performance-based recommendations
            performance_summary = self._create_performance_summary(all_regime_results)
            overall_performance = performance_summary.get('overall_performance', 0.0)
            
            if overall_performance < 0.6:
                recommendations.append("Overall performance is below optimal - consider parameter tuning")
            elif overall_performance > 0.8:
                recommendations.append("Excellent performance - consider expanding to similar regimes")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating regime recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _calculate_saving_metadata(self, saved_components: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate saving metadata.
        
        Args:
            saved_components: Saved components
            
        Returns:
            Saving metadata
        """
        try:
            return {
                'total_components_saved': len(saved_components),
                'component_types': list(saved_components.keys()),
                'saving_timestamp': datetime.now().isoformat(),
                'saving_status': 'completed'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating saving metadata: {e}")
            return {'saving_status': 'error'}
    
    async def _save_regime_saving_results(
        self,
        saving_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save saving results for a specific regime.
        
        Args:
            saving_results: Saving results
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
            saving_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_saving_regime_{regime_id}.json'
            
            with open(saving_path, 'w') as f:
                json.dump(saving_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved saving results for regime {regime_id}: {saving_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving saving results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_saving_step')
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
    """Run the enhanced per-regime saving step.
    
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
    logger.info("🚀 Starting Step 21: Per-Regime Saving")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_saving'] = True
    
    # Initialize and run the per-regime saving step
    step = PerRegimeSavingStep(config)
    
    success = await step.execute_per_regime_saving(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 21: Per-Regime Saving completed successfully")
    else:
        logger.error("❌ Step 21: Per-Regime Saving failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime saving step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime saving result: {success}')
        
    asyncio.run(test())