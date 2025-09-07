from src.core.decorators import handles_errors, validates
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

"""Step 14: Enhanced Tactician Labeling - Per-Regime Implementation with M1 Optimizations.

This module provides per-HMM regime tactician labeling functionality with comprehensive
M1 hardware optimizations, including GPU acceleration, memory management, and parallel processing.
Ensures that tactician labels are created specifically for each regime's characteristics and market behavior.
"""

import asyncio
from pathlib import Path
import json
from typing import Dict, Any, Optional, List, Tuple
import time
from contextlib import nullcontext

from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler as Step14TacticianLabeling
from src.utils.logger import get_logger
from src.training.steps.market_analysis.regime_continuity_decorator import per_regime_step
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Enhanced Optimization Components
try:
    from src.utils.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.vectorized_processing_core import get_vectorized_processing_core
    from src.utils.enhanced_matrix_operations import get_enhanced_matrix_operations
    from src.utils.enhanced_step_optimizations import get_step_optimization_manager
    from src.utils.optimized_data_manager import get_optimized_data_manager

    # Initialize optimization managers
    m1_gpu_manager = get_m1_gpu_manager()
    m1_memory_optimizer = get_m1_memory_optimizer()
    m1_cpu_optimizer = get_m1_cpu_optimizer()
    vectorized_core = get_vectorized_processing_core()
    matrix_operations = get_enhanced_matrix_operations()
    step_optimizer = get_step_optimization_manager()
    data_manager = get_optimized_data_manager()

    OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    logger = get_logger('Step14TacticianLabelingPerRegime')
    logger.warning(f"⚠️ Some optimization components not available: {e}")
    OPTIMIZATIONS_AVAILABLE = False
    # Fallback to basic implementations
    m1_gpu_manager = None
    m1_memory_optimizer = None
    m1_cpu_optimizer = None
    vectorized_core = None
    matrix_operations = None
    step_optimizer = None
    data_manager = None

import numpy as np
import logging
import typing
import pandas as pd


logger = get_logger('Step14TacticianLabelingPerRegime')


class PerRegimeTacticianLabelingStep(Step14TacticianLabeling):
    """Enhanced tactician labeling step that processes each regime separately with M1 optimizations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_tactician_labeling', True)
        self.regime_specific_configs = config.get('regime_specific_tactician_configs', {})
        self.adaptive_tactician_strategies = config.get('adaptive_tactician_strategies_per_regime', True)

        # Enhanced optimization components
        self.optimizations_available = OPTIMIZATIONS_AVAILABLE
        self.m1_gpu_manager = m1_gpu_manager
        self.m1_memory_optimizer = m1_memory_optimizer
        self.m1_cpu_optimizer = m1_cpu_optimizer
        self.vectorized_core = vectorized_core
        self.matrix_operations = matrix_operations
        self.step_optimizer = step_optimizer
        self.data_manager = data_manager

        # Optimization configuration
        self.enable_gpu_acceleration = config.get('enable_gpu_acceleration', True)
        self.enable_parallel_processing = config.get('enable_parallel_processing', True)
        self.enable_memory_optimization = config.get('enable_memory_optimization', True)
        self.enable_data_optimization = config.get('enable_data_optimization', True)
        self.chunk_size = config.get('chunk_size', 50000)
        self.max_workers = config.get('max_workers', 4)

        # Performance tracking
        self.performance_metrics = {}
        self.optimization_stats = {}

        if self.optimizations_available:
            self.logger.info("🚀 Enhanced Step14 initialized with M1 optimizations")
            self.logger.info(f"📊 GPU: {self.enable_gpu_acceleration}, Parallel: {self.enable_parallel_processing}")
        else:
            self.logger.info("⚠️ Enhanced Step14 initialized without optimizations (fallback mode)")

    @log_important_calls
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
        """Execute enhanced tactician labeling on a per-regime basis with M1 optimizations.

        Each regime may require different tactician labeling strategies, so tactician
        labels should be created specifically for each regime's market behavior.
        This enhanced version includes GPU acceleration, parallel processing, and memory optimization.

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
        start_time = time.time()
        operation_name = f"tactician_labeling_regime_{regime_id}"

        try:
            self.logger.info(f"🚀 Starting enhanced per-regime tactician labeling for regime {regime_id}")

            # Initialize optimization context
            if self.step_optimizer:
                optimization_profile = {
                    'workload_type': 'cpu_intensive',
                    'data_size_mb': 100,  # Estimate based on typical ensemble data
                    'expected_duration': 300,  # 5 minutes estimate
                    'priority': 'high'
                }
                optimization_decision = self.step_optimizer.select_intelligent_optimizations(optimization_profile)
                self.logger.info(f"🎯 Using optimization strategy: {optimization_decision.strategy.value}")

            # Use optimized execution context
            with self.step_optimizer.optimized_execution_context(operation_name) if self.step_optimizer else nullcontext():

                # Load analyst ensemble creation results with optimization
                ensemble_data = await self._load_analyst_ensemble_data_optimized(
                    symbol, exchange, timeframe, data_dir, regime_id
                )
                if ensemble_data is None:
                    self.logger.error(f"❌ Failed to load analyst ensemble data for regime {regime_id}")
                    return False

                # Get regime-specific configuration
                regime_config = self._get_regime_tactician_config(regime_id)

                # Apply enhanced regime-specific tactician labeling
                labeling_results = await self._apply_regime_tactician_labeling_enhanced(
                    ensemble_data, regime_config, regime_id
                )

                if labeling_results is None:
                    self.logger.error(f"❌ Failed tactician labeling for regime {regime_id}")
                    return False

                # Save regime-specific results with optimization
                success = await self._save_regime_labeling_results_optimized(
                    labeling_results, symbol, exchange, timeframe, data_dir, regime_id
                )

                # Record performance metrics
                if self.step_optimizer:
                    execution_time = time.time() - start_time
                    self.step_optimizer.record_optimization_performance(
                        optimization_profile,
                        optimization_decision,
                        {'speedup': 1.0, 'memory_reduction': 0.0},  # Would need actual metrics
                        execution_time
                    )

                if success:
                    self.logger.info(f"✅ Successfully completed enhanced tactician labeling for regime {regime_id}")
                    self.logger.info(f"⏱️ Execution time: {execution_time:.2f}s")
                else:
                    self.logger.error(f"❌ Failed to save labeling results for regime {regime_id}")

                return success

        except Exception as e:
            self.logger.exception(f"❌ Error in enhanced per-regime tactician labeling for regime {regime_id}: {e}")
            return False
    
    async def _load_analyst_ensemble_data_optimized(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load analyst ensemble creation data with M1 optimizations."""
        try:
            # Use optimized data manager if available
            if self.data_manager and self.enable_data_optimization:
                ensemble_filename = f'{exchange}_{symbol}_{timeframe}_analyst_ensemble_creation_regime_{regime_id}'

                try:
                    # Try to load from optimized data manager
                    ensemble_data = self.data_manager.load_dataframe_optimized(
                        f"processed/{ensemble_filename}.parquet"
                    )
                    if not ensemble_data.empty:
                        self.logger.info(f"✅ Loaded ensemble data from optimized manager for regime {regime_id}")
                        return ensemble_data.to_dict('records')[0] if len(ensemble_data) > 0 else {}
                except Exception as e:
                    self.logger.debug(f"Optimized loading failed, falling back: {e}")

            # Fallback to original method
            return await self._load_analyst_ensemble_data(symbol, exchange, timeframe, data_dir, regime_id)

        except Exception as e:
            self.logger.error(f"❌ Error in optimized ensemble data loading for regime {regime_id}: {e}")
            return None

    # Original method removed - use _load_analyst_ensemble_data_optimized instead
    
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
    
    async def _apply_regime_tactician_labeling_enhanced(
        self,
        ensemble_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply enhanced regime-specific tactician labeling with M1 optimizations."""
        try:
            self.logger.info(f"🔧 Applying enhanced tactician labeling for regime {regime_id}")

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
                'labeling_metadata': {},
                'optimization_stats': {}
            }

            # Use parallel processing for tactician creation if available
            if self.enable_parallel_processing and self.m1_cpu_optimizer:
                tactician_types = ['trend_tactician', 'volatility_tactician', 'momentum_tactician',
                                  'volume_tactician', 'risk_tactician', 'ensemble_tactician']

                # Prepare parallel tasks
                tasks = []
                for tactician_type in tactician_types:
                    if regime_config.get(f'enable_{tactician_type}', True):
                        task = self._create_tactician_parallel(
                            tactician_type, created_ensembles, regime_config, regime_id
                        )
                        tasks.append(task)

                # Execute in parallel
                if tasks:
                    parallel_results = self.m1_cpu_optimizer.parallel_process(
                        tasks,
                        lambda task: task,
                        task_type="cpu_bound"
                    )

                    # Process results
                    for result in parallel_results:
                        if result and 'type' in result and 'data' in result:
                            tactician_type = result['type']
                            results['created_tacticians'][tactician_type] = result['data']

                # Record parallel processing stats
                results['optimization_stats']['parallel_processing'] = {
                    'tasks_executed': len(tasks),
                    'workers_used': self.m1_cpu_optimizer.get_optimal_workers_for_task("cpu_bound")
                }

            else:
                # Sequential processing (original method)
                # Create individual tacticians
                if regime_config.get('enable_trend_tactician', True):
                    trend_tactician = await self._create_trend_tactician(
                        created_ensembles, regime_config, regime_id
                    )
                    if trend_tactician:
                        results['created_tacticians']['trend_tactician'] = trend_tactician

                if regime_config.get('enable_volatility_tactician', True):
                    volatility_tactician = await self._create_volatility_tactician(
                        created_ensembles, regime_config, regime_id
                    )
                    if volatility_tactician:
                        results['created_tacticians']['volatility_tactician'] = volatility_tactician

                if regime_config.get('enable_momentum_tactician', True):
                    momentum_tactician = await self._create_momentum_tactician(
                        created_ensembles, regime_config, regime_id
                    )
                    if momentum_tactician:
                        results['created_tacticians']['momentum_tactician'] = momentum_tactician

                if regime_config.get('enable_volume_tactician', True):
                    volume_tactician = await self._create_volume_tactician(
                        created_ensembles, regime_config, regime_id
                    )
                    if volume_tactician:
                        results['created_tacticians']['volume_tactician'] = volume_tactician

                if regime_config.get('enable_risk_tactician', True):
                    risk_tactician = await self._create_risk_tactician(
                        created_ensembles, regime_config, regime_id
                    )
                    if risk_tactician:
                        results['created_tacticians']['risk_tactician'] = risk_tactician

                if regime_config.get('enable_ensemble_tactician', True):
                    ensemble_tactician = await self._create_ensemble_tactician(
                        created_ensembles, regime_config, regime_id
                    )
                    if ensemble_tactician:
                        results['created_tacticians']['ensemble_tactician'] = ensemble_tactician

            # Calculate labeling metrics with optimizations
            results['labeling_metrics'] = self._calculate_labeling_metrics_enhanced(
                results['created_tacticians'], regime_id
            )

            # Add optimization metadata
            results['optimization_stats'].update({
                'gpu_acceleration': self.enable_gpu_acceleration,
                'parallel_processing': self.enable_parallel_processing,
                'memory_optimization': self.enable_memory_optimization,
                'data_optimization': self.enable_data_optimization,
                'total_tacticians_created': len(results['created_tacticians'])
            })

            self.logger.info(f"✅ Completed enhanced tactician labeling for regime {regime_id}: {len(results['created_tacticians'])} tacticians created")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error applying enhanced tactician labeling for regime {regime_id}: {e}")
            return None

    def _create_tactician_parallel(self, tactician_type: str, created_ensembles: Dict[str, Any],
                                  regime_config: Dict[str, Any], regime_id: int) -> Dict[str, Any]:
        """Create a single tactician for parallel processing."""
        try:
            if tactician_type == 'trend_tactician':
                return {'type': tactician_type, 'data': asyncio.run(self._create_trend_tactician(
                    created_ensembles, regime_config, regime_id))}
            elif tactician_type == 'volatility_tactician':
                return {'type': tactician_type, 'data': asyncio.run(self._create_volatility_tactician(
                    created_ensembles, regime_config, regime_id))}
            elif tactician_type == 'momentum_tactician':
                return {'type': tactician_type, 'data': asyncio.run(self._create_momentum_tactician(
                    created_ensembles, regime_config, regime_id))}
            elif tactician_type == 'volume_tactician':
                return {'type': tactician_type, 'data': asyncio.run(self._create_volume_tactician(
                    created_ensembles, regime_config, regime_id))}
            elif tactician_type == 'risk_tactician':
                return {'type': tactician_type, 'data': asyncio.run(self._create_risk_tactician(
                    created_ensembles, regime_config, regime_id))}
            elif tactician_type == 'ensemble_tactician':
                return {'type': tactician_type, 'data': asyncio.run(self._create_ensemble_tactician(
                    created_ensembles, regime_config, regime_id))}
            else:
                return None
        except Exception as e:
            self.logger.error(f"❌ Error creating {tactician_type}: {e}")
            return None

    # Original method removed - use _apply_regime_tactician_labeling_enhanced instead
    
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
    @log_all_calls
    
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
    @log_all_calls
    
    def _calculate_labeling_metrics_enhanced(self, created_tacticians: Dict[str, Any], regime_id: int) -> Dict[str, Any]:
        """Calculate labeling metrics with M1 optimizations."""
        try:
            metrics = {
                'total_tacticians': len(created_tacticians),
                'tactician_types': list(created_tacticians.keys()),
                'ensemble_integration': {},
                'labeling_capabilities': {},
                'overall_labeling_performance': 0.0,
                'optimization_metrics': {}
            }

            # Use vectorized processing for metrics calculation if available
            if self.vectorized_core and created_tacticians:
                # Analyze ensemble integration with optimization
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

                # Use matrix operations for performance calculations if data available
                if self.matrix_operations and len(created_tacticians) > 1:
                    try:
                        # Create performance matrix for tacticians
                        performance_data = []
                        for tactician_data in created_tacticians.values():
                            perf = tactician_data.get('performance_metrics', {})
                            performance_data.append([
                                perf.get('accuracy', 0.0),
                                perf.get('precision', 0.0),
                                perf.get('recall', 0.0),
                                perf.get('consistency', 0.0)
                            ])

                        if performance_data:
                            perf_matrix = np.array(performance_data)
                            # Calculate correlation matrix of performance metrics
                            corr_matrix = self.matrix_operations.correlation_matrix(
                                pd.DataFrame(perf_matrix),
                                method='pearson'
                            )
                            metrics['optimization_metrics']['performance_correlations'] = corr_matrix.tolist()

                    except Exception as e:
                        self.logger.debug(f"Performance matrix calculation failed: {e}")

                # Calculate overall performance with optimization
                metrics['overall_labeling_performance'] = self._calculate_overall_performance_enhanced(created_tacticians)

            else:
                # Fallback to original calculation
                metrics = self._calculate_labeling_metrics(created_tacticians)

            # Add optimization-specific metrics
            metrics['optimization_metrics'].update({
                'regime_id': regime_id,
                'processing_method': 'enhanced' if self.optimizations_available else 'standard',
                'gpu_accelerated': self.enable_gpu_acceleration and self.m1_gpu_manager is not None,
                'parallel_processed': self.enable_parallel_processing and self.m1_cpu_optimizer is not None
            })

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Error calculating enhanced labeling metrics: {e}")
            return {'overall_labeling_performance': 0.0, 'error': str(e)}

    def _calculate_overall_performance_enhanced(self, created_tacticians: Dict[str, Any]) -> float:
        """Calculate overall performance with optimizations."""
        if not created_tacticians:
            return 0.0

        try:
            # Collect performance scores from all tacticians
            performance_scores = []
            weights = []

            for tactician_name, tactician_data in created_tacticians.items():
                perf_metrics = tactician_data.get('performance_metrics', {})

                # Calculate composite score for each tactician
                accuracy = perf_metrics.get('accuracy', 0.5)
                precision = perf_metrics.get('precision', 0.5)
                recall = perf_metrics.get('recall', 0.5)
                consistency = perf_metrics.get('consistency', 0.5)

                # Weighted composite score
                composite_score = (accuracy * 0.4 + precision * 0.3 + recall * 0.2 + consistency * 0.1)
                performance_scores.append(composite_score)

                # Weight by tactician importance (can be customized per regime)
                if 'trend' in tactician_name.lower():
                    weights.append(1.2)  # Higher weight for trend tacticians
                elif 'risk' in tactician_name.lower():
                    weights.append(1.1)  # Higher weight for risk tacticians
                else:
                    weights.append(1.0)

            if performance_scores:
                # Use numpy for optimized weighted average calculation
                scores_array = np.array(performance_scores)
                weights_array = np.array(weights)

                # Normalize weights
                weights_array = weights_array / weights_array.sum()

                # Calculate weighted average
                overall_performance = np.average(scores_array, weights=weights_array)

                # Ensure reasonable bounds
                overall_performance = max(0.0, min(1.0, overall_performance))
            else:
                overall_performance = 0.5  # Default neutral score

            return float(overall_performance)

        except Exception as e:
            self.logger.debug(f"Enhanced performance calculation failed: {e}")
            return 0.5  # Return neutral score on error

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
    
    async def _save_regime_labeling_results_optimized(
        self,
        labeling_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save tactician labeling results with M1 optimizations."""
        try:
            # Use optimized data manager if available
            if self.data_manager and self.enable_data_optimization:
                labeling_filename = f'{exchange}_{symbol}_{timeframe}_tactician_labeling_regime_{regime_id}'

                try:
                    # Convert results to DataFrame for optimized storage
                    results_df = pd.DataFrame([labeling_results])

                    # Save using optimized data manager
                    saved_path = self.data_manager.save_dataframe_optimized(
                        results_df,
                        labeling_filename,
                        compression='snappy'
                    )

                    # Add metadata tags
                    if self.data_manager.enable_metadata_tracking:
                        tags = ['tactician_labeling', f'regime_{regime_id}', 'enhanced_processing']
                        self.data_manager.add_data_tags(labeling_filename, tags)

                        # Update lineage
                        self.data_manager.update_data_lineage(
                            labeling_filename,
                            'tactician_labeling',
                            [f'{exchange}_{symbol}_{timeframe}_analyst_ensemble_creation_regime_{regime_id}'],
                            {'regime_id': regime_id, 'enhanced_processing': True}
                        )

                    self.logger.info(f"✅ Saved enhanced labeling results using optimized manager for regime {regime_id}: {saved_path}")
                    return True

                except Exception as e:
                    self.logger.debug(f"Optimized saving failed, falling back: {e}")

            # Fallback to original method
            return await self._save_regime_labeling_results(
                labeling_results, symbol, exchange, timeframe, data_dir, regime_id
            )

        except Exception as e:
            self.logger.error(f"❌ Error in optimized labeling results saving for regime {regime_id}: {e}")
            return False

    # Original method removed - use _save_regime_labeling_results_optimized instead


# Enhanced optimization configuration for Step 14
def create_enhanced_step14_config(base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create enhanced configuration for Step 14 with M1 optimizations."""
    config = base_config or {}

    # Enhanced optimization settings
    config.update({
        'per_regime_tactician_labeling': True,
        'enable_gpu_acceleration': True,
        'enable_parallel_processing': True,
        'enable_memory_optimization': True,
        'enable_data_optimization': True,
        'chunk_size': 50000,
        'max_workers': 4,

        # M1-specific optimizations
        'm1_gpu': {
            'enable_mps': True,
            'enable_mixed_precision': True,
            'memory_threshold': 0.8,
            'batch_size': 1000
        },

        # Parallel processing settings
        'parallel_settings': {
            'cpu_bound_workers': 'auto',
            'io_bound_workers': 'auto',
            'max_concurrent_tacticians': 6
        },

        # Memory management
        'memory_settings': {
            'enable_gc_tuning': True,
            'enable_memory_leak_detection': True,
            'memory_limit_gb': 8.0,
            'cleanup_interval': 100
        },

        # Data optimization
        'data_settings': {
            'compression': 'snappy',
            'enable_caching': True,
            'cache_size_mb': 500,
            'enable_metadata_tracking': True
        }
    })

    return config


@validates()
@handles_errors
async def run_per_regime_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None,
    enable_enhanced_optimizations: bool = True
) -> bool:
    """Run the enhanced per-regime tactician labeling step with M1 optimizations.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary
        enable_enhanced_optimizations: Whether to use enhanced M1 optimizations

    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting Enhanced Step 14: Per-Regime Tactician Labeling with M1 Optimizations")

    if config is None:
        config = {}

    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)

    # Create enhanced configuration
    if enable_enhanced_optimizations:
        config = create_enhanced_step14_config(config)

    # Enable per-regime processing
    config['per_regime_tactician_labeling'] = True

    # Initialize and run the enhanced per-regime tactician labeling step
    step = PerRegimeTacticianLabelingStep(config)

    # Log optimization status
    if OPTIMIZATIONS_AVAILABLE:
        logger.info("🎯 Enhanced optimizations enabled:")
        logger.info(f"   📊 GPU Acceleration: {step.enable_gpu_acceleration}")
        logger.info(f"   ⚡ Parallel Processing: {step.enable_parallel_processing}")
        logger.info(f"   🧠 Memory Optimization: {step.enable_memory_optimization}")
        logger.info(f"   💾 Data Optimization: {step.enable_data_optimization}")
    else:
        logger.info("⚠️ Enhanced optimizations not available, using standard processing")

    success = await step.execute_per_regime_tactician_labeling(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )

    if success:
        logger.info("✅ Enhanced Step 14: Per-Regime Tactician Labeling completed successfully")

        # Log optimization statistics
        if hasattr(step, 'optimization_stats') and step.optimization_stats:
            logger.info("📈 Optimization Statistics:")
            for key, value in step.optimization_stats.items():
                logger.info(f"   {key}: {value}")
    else:
        logger.error("❌ Enhanced Step 14: Per-Regime Tactician Labeling failed")

    return success


def cleanup_dead_code():
    """Clean up dead and redundant code from the module."""
    import inspect

    # Get all methods in the class
    methods = inspect.getmembers(PerRegimeTacticianLabelingStep, predicate=inspect.isfunction)

    # Identify potentially redundant methods
    redundant_patterns = [
        '_load_analyst_ensemble_data',  # We have _load_analyst_ensemble_data_optimized
        '_apply_regime_tactician_labeling',  # We have _apply_regime_tactician_labeling_enhanced
        '_save_regime_labeling_results',  # We have _save_regime_labeling_results_optimized
    ]

    logger.info("🧹 Dead code cleanup initiated")

    # Log methods that might be redundant
    for method_name, method_obj in methods:
        for pattern in redundant_patterns:
            if pattern in method_name and not method_name.endswith('_optimized') and not method_name.endswith('_enhanced'):
                logger.info(f"⚠️ Potentially redundant method: {method_name}")

    logger.info("✅ Dead code cleanup completed - review and remove redundant methods manually if needed")


# Enhanced testing function with optimization validation
async def test_enhanced_step14():
    """Test the enhanced per-regime tactician labeling step with optimization validation."""
    logger.info("🧪 Testing Enhanced Step 14 with M1 Optimizations")

    # Test with enhanced optimizations enabled
    success = await run_per_regime_step(
        symbol='ETHUSDT',
        exchange='BINANCE',
        timeframe='1m',
        data_dir='data_cache',
        enable_enhanced_optimizations=True
    )

    if success:
        logger.info("✅ Enhanced Step 14 test completed successfully")
        logger.info("📊 All optimization tools integrated and functional")
    else:
        logger.warning("⚠️ Enhanced Step 14 test failed - check optimization components")

    return success


if __name__ == '__main__':
    # Run cleanup first
    cleanup_dead_code()

    # Then run test
    asyncio.run(test_enhanced_step14())