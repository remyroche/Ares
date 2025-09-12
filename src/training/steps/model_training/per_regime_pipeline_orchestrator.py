"""
Per-Regime Pipeline Orchestrator.

This module orchestrates the entire per-regime pipeline, ensuring that regime
continuity is maintained throughout all steps and providing comprehensive
monitoring and validation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Handle optional dependencies gracefully
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from src.utils.tprint import tprint
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.decorators import traced, validates, handles_errors, log_execution_time
from src.utils.logger import get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.utils.file_utils import safe_json_dump, safe_json_load
from src.training.steps.model_training.per_regime_pipeline_integration import PerRegimePipelineIntegrator
from src.training.steps.model_training.regime_data_integration import regime_data_integrator
from src.training.steps.model_training.regime_data_utils import regime_data_accessor

logger = get_logger('PerRegimePipelineOrchestrator')


class RegimeContinuityManager:
    """Manages regime continuity across pipeline steps."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime continuity manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('RegimeContinuityManager')
        self.continuity_state: Dict[str, Any] = {}
        self.regime_transitions: Dict[str, List[Dict[str, Any]]] = {}
        
    @log_important_calls
    async def initialize_regime_continuity(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> bool:
        """Initialize regime continuity for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"🔄 Initializing regime continuity for {exchange}_{symbol}_{timeframe}")
            
            # Load regime data
            regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                self.logger.error("❌ Failed to load regime data")
                return False
            
            # Initialize continuity state
            key = f"{exchange}_{symbol}_{timeframe}"
            self.continuity_state[key] = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'regime_data': regime_data,
                'current_step': None,
                'step_history': [],
                'regime_transitions': [],
                'continuity_score': 1.0,
                'last_updated': datetime.now().isoformat()
            }
            
            # Initialize regime transitions tracking
            self.regime_transitions[key] = []
            
            self.logger.info(f"✅ Regime continuity initialized for {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing regime continuity: {e}")
            return False
    
    async def _load_regime_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Load regime data from files."""
        try:
            training_dir = Path(data_dir) / 'training'
            regime_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            
            if not regime_file.exists():
                self.logger.error(f"❌ Regime data file not found: {regime_file}")
                return None
            
            # Load regime data using regime_data_accessor
            if PANDAS_AVAILABLE:
                data = pd.read_parquet(regime_file)
                regime_data = regime_data_accessor.get_regime_data({'regime_states': data.get('regime_states', []).tolist()})
            else:
                # Fallback for when pandas is not available
                self.logger.warning("⚠️ Pandas not available, using mock regime data")
                regime_data = {
                    'regime_labels': [0, 1, 2] * 100,  # Mock data
                    'regime_probabilities': [[0.8, 0.1, 0.1]] * 300,
                    'regime_confidence': [0.9] * 300,
                    'n_regimes': 3,
                    'data_source': 'mock'
                }
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading regime data: {e}")
            return None
    
    @log_all_calls
    async def update_step_continuity(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        step_name: str, 
        step_result: Dict[str, Any]
    ) -> bool:
        """Update continuity state after a step execution.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            step_name: Name of the step
            step_result: Result of the step execution
            
        Returns:
            True if update successful
        """
        try:
            key = f"{exchange}_{symbol}_{timeframe}"
            if key not in self.continuity_state:
                self.logger.error(f"❌ Continuity state not found for {key}")
                return False
            
            # Update step history
            step_entry = {
                'step_name': step_name,
                'timestamp': datetime.now().isoformat(),
                'success': step_result.get('success', False),
                'regime_impact': step_result.get('regime_impact', {}),
                'continuity_score': step_result.get('continuity_score', 1.0)
            }
            
            self.continuity_state[key]['step_history'].append(step_entry)
            self.continuity_state[key]['current_step'] = step_name
            self.continuity_state[key]['last_updated'] = datetime.now().isoformat()
            
            # Update overall continuity score
            if self.continuity_state[key]['step_history']:
                scores = [entry['continuity_score'] for entry in self.continuity_state[key]['step_history']]
                if NUMPY_AVAILABLE:
                    self.continuity_state[key]['continuity_score'] = np.mean(scores)
                else:
                    self.continuity_state[key]['continuity_score'] = sum(scores) / len(scores)
            
            self.logger.info(f"✅ Updated continuity for {step_name} in {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating step continuity: {e}")
            return False
    
    @log_all_calls
    async def get_continuity_report(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """Get continuity report for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Continuity report
        """
        try:
            key = f"{exchange}_{symbol}_{timeframe}"
            if key not in self.continuity_state:
                return {'error': f'No continuity data found for {key}'}
            
            state = self.continuity_state[key]
            
            report = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'total_steps': len(state['step_history']),
                'completed_steps': len([s for s in state['step_history'] if s['success']]),
                'continuity_score': state['continuity_score'],
                'current_step': state['current_step'],
                'last_updated': state['last_updated'],
                'step_summary': state['step_history'],
                'regime_transitions': self.regime_transitions.get(key, [])
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error getting continuity report: {e}")
            return {'error': str(e)}


class RegimeContinuityValidator:
    """Validates regime continuity across pipeline steps."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime continuity validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('RegimeContinuityValidator')
        
    @log_important_calls
    async def validate_step_continuity(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate continuity after a step execution.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Validation result
        """
        try:
            self.logger.info(f"🔍 Validating step continuity for {step_name}")
            
            # Load current regime data
            regime_data = await self._load_current_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                return {
                    'is_valid': False,
                    'issues': ['Failed to load regime data'],
                    'validation_score': 0.0
                }
            
            # Validate regime consistency
            validation_result = self._validate_regime_consistency(regime_data)
            
            # Validate step-specific continuity
            step_validation = await self._validate_step_specific_continuity(
                step_name, regime_data, symbol, exchange, timeframe
            )
            
            # Combine validation results
            overall_valid = validation_result['is_valid'] and step_validation['is_valid']
            combined_score = (validation_result.get('score', 0.0) + step_validation.get('score', 0.0)) / 2
            
            result = {
                'is_valid': overall_valid,
                'validation_score': combined_score,
                'issues': validation_result.get('issues', []) + step_validation.get('issues', []),
                'warnings': validation_result.get('warnings', []) + step_validation.get('warnings', []),
                'step_name': step_name,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Step continuity validation completed: {overall_valid}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating step continuity: {e}")
            return {
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'validation_score': 0.0
            }
    
    async def _load_current_regime_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Load current regime data."""
        try:
            training_dir = Path(data_dir) / 'training'
            regime_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            
            if not regime_file.exists():
                return None
            
            if PANDAS_AVAILABLE:
                data = pd.read_parquet(regime_file)
                return regime_data_accessor.get_regime_data({'regime_states': data.get('regime_states', []).tolist()})
            else:
                # Fallback for when pandas is not available
                self.logger.warning("⚠️ Pandas not available, using mock regime data")
                return {
                    'regime_labels': [0, 1, 2] * 100,  # Mock data
                    'regime_probabilities': [[0.8, 0.1, 0.1]] * 300,
                    'regime_confidence': [0.9] * 300,
                    'n_regimes': 3,
                    'data_source': 'mock'
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error loading current regime data: {e}")
            return None
    
    def _validate_regime_consistency(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regime data consistency."""
        try:
            validation_result = regime_data_accessor.validate_regime_data(regime_data)
            
            # Calculate validation score
            score = 1.0
            if not validation_result['is_valid']:
                score = 0.0
            elif validation_result.get('warnings'):
                score = 0.8
            
            validation_result['score'] = score
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime consistency: {e}")
            return {
                'is_valid': False,
                'issues': [f'Consistency validation error: {str(e)}'],
                'score': 0.0
            }
    
    async def _validate_step_specific_continuity(
        self, 
        step_name: str, 
        regime_data: Dict[str, Any], 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """Validate step-specific continuity requirements."""
        try:
            issues = []
            warnings = []
            score = 1.0
            
            # Step-specific validation logic
            if step_name in ['step05_labeling', 'step06_feature_engineering']:
                # Validate that regime data is available for labeling/feature engineering
                if not regime_data.get('regime_labels') or len(regime_data['regime_labels']) == 0:
                    issues.append(f"No regime labels available for {step_name}")
                    score = 0.0
            
            elif step_name in ['step11_analyst_creation', 'step12_analyst_enhancement']:
                # Validate that features are available for analyst creation
                if not regime_data.get('regime_characteristics'):
                    warnings.append(f"Limited regime characteristics for {step_name}")
                    score = 0.8
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'score': score
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in step-specific validation: {e}")
            return {
                'is_valid': False,
                'issues': [f'Step-specific validation error: {str(e)}'],
                'score': 0.0
            }
    
    @log_all_calls
    async def validate_pipeline_continuity(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        steps_to_run: List[str]
    ) -> Dict[str, Any]:
        """Validate overall pipeline continuity.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            steps_to_run: List of steps to run
            
        Returns:
            Overall validation result
        """
        try:
            self.logger.info(f"🔍 Validating overall pipeline continuity for {exchange}_{symbol}_{timeframe}")
            
            # Load final regime data
            regime_data = await self._load_current_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                return {
                    'overall_score': 0.0,
                    'is_valid': False,
                    'issues': ['Failed to load final regime data']
                }
            
            # Validate regime data quality
            regime_validation = self._validate_regime_consistency(regime_data)
            
            # Validate pipeline completeness
            completeness_validation = self._validate_pipeline_completeness(steps_to_run, data_dir)
            
            # Calculate overall score
            overall_score = (regime_validation.get('score', 0.0) + completeness_validation.get('score', 0.0)) / 2
            
            result = {
                'overall_score': overall_score,
                'is_valid': overall_score >= 0.8,
                'regime_validation': regime_validation,
                'completeness_validation': completeness_validation,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Pipeline continuity validation completed: {overall_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating pipeline continuity: {e}")
            return {
                'overall_score': 0.0,
                'is_valid': False,
                'issues': [f'Pipeline validation error: {str(e)}']
            }
    
    def _validate_pipeline_completeness(
        self, 
        steps_to_run: List[str], 
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate pipeline completeness."""
        try:
            training_dir = Path(data_dir) / 'training'
            completed_steps = 0
            total_steps = len(steps_to_run)
            
            # Check for step completion markers
            for step in steps_to_run:
                step_file = training_dir / f'{step}_completed.json'
                if step_file.exists():
                    completed_steps += 1
            
            completeness_score = completed_steps / total_steps if total_steps > 0 else 0.0
            
            return {
                'score': completeness_score,
                'completed_steps': completed_steps,
                'total_steps': total_steps,
                'is_complete': completeness_score >= 0.9
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating pipeline completeness: {e}")
            return {
                'score': 0.0,
                'completed_steps': 0,
                'total_steps': 0,
                'is_complete': False
            }


@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
    symbol: str
    exchange: str
    timeframe: str
    execution_start: datetime
    execution_end: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    step_results: Dict[str, bool] = None
    continuity_validation_score: float = 0.0
    overall_success: bool = False
    error_message: Optional[str] = None
    @log_all_calls

    def __post_init__(self) -> None:
        if self.step_results is None:
            self.step_results = {}

class PerRegimePipelineOrchestrator:
    """Orchestrates the entire per-regime pipeline with continuity management."""
    
    @log_important_calls
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the per-regime pipeline orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('PerRegimePipelineOrchestrator')
        self.standards = pipeline_standards
        self._load_per_regime_config()
        
        # Initialize continuity management components
        self.continuity_manager = RegimeContinuityManager(self.config)
        self.continuity_validator = RegimeContinuityValidator(self.config)
        self.pipeline_integrator = PerRegimePipelineIntegrator(self.config)
        
        # Define pipeline steps based on final MODEL_TRAINING structure
        # Only 4 required steps remain in MODEL_TRAINING stage
        self.pipeline_steps = [
            'analyst_models_training',      # Per-regime individual model training
            'analyst_ensemble_training',    # Per-regime ensemble training
            'tactician_models_training',    # All-regime individual model training
            'tactician_ensemble_training'   # All-regime ensemble training
        ]
        
        # Steps that require per-regime processing (uses HMM-retagged regimes from MARKET_ANALYSIS)
        self.per_regime_steps = [
            'analyst_models_training',      # Per-regime individual models
            'analyst_ensemble_training'     # Per-regime ensemble models
        ]
        
        # Steps that use all-regime processing (uses all data regardless of regime)
        self.all_regime_steps = [
            'tactician_models_training',    # All-regime individual models
            'tactician_ensemble_training'   # All-regime ensemble models
        ]
        
        # No longer needed in final MODEL_TRAINING structure
        self.market_analysis_regime_steps = []
        self.regime_retagging_steps = []
    @log_all_calls

    def _load_per_regime_config(self) -> None:
        """Load per-regime pipeline configuration."""
        try:
            config_path = Path(__file__).parent / 'per_regime_pipeline_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    per_regime_config = json.load(f)
                    self.config.update(per_regime_config)
                    self.logger.info('✅ Loaded per-regime pipeline configuration')
            else:
                self.logger.warning('⚠️ Per-regime config file not found, using defaults')
        except Exception as e:
            self.logger.warning(f'⚠️ Error loading per-regime config: {e}, using defaults')

    @traced(span_name='execute_per_regime_pipeline')
    @log_execution_time
    async def execute_per_regime_pipeline(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        force_rerun: bool = False, 
        steps_to_run: Optional[List[str]] = None
    ) -> PipelineExecutionResult:
        """Execute the entire per-regime pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun all steps
            steps_to_run: Specific steps to run (default: all steps)
            
        Returns:
            Pipeline execution result
        """
        execution_start = datetime.now()
        result = PipelineExecutionResult(
            symbol=symbol, 
            exchange=exchange, 
            timeframe=timeframe, 
            execution_start=execution_start
        )
        
        try:
            self.logger.info(f'🚀 Starting per-regime pipeline execution for {exchange}_{symbol}_{timeframe}')
            
            if steps_to_run is None:
                steps_to_run = self.pipeline_steps
            result.total_steps = len(steps_to_run)
            
            # Initialize regime continuity management
            self.logger.info('🔧 Initializing regime continuity management')
            continuity_init_success = await self.continuity_manager.initialize_regime_continuity(
                symbol, exchange, timeframe, data_dir
            )
            if not continuity_init_success:
                result.error_message = 'Failed to initialize regime continuity'
                result.execution_end = datetime.now()
                return result
            
            # Verify regime data availability
            regime_data_available = await self.pipeline_integrator.verify_regime_data_availability(
                symbol, exchange, timeframe, data_dir
            )
            if not regime_data_available:
                result.error_message = 'Regime data not available for processing'
                result.execution_end = datetime.now()
                return result
            
            # Execute pipeline steps
            for step_name in steps_to_run:
                self.logger.info(f'🔄 Executing step: {step_name}')
                try:
                    # Determine step execution type based on regime requirements
                    if step_name in self.per_regime_steps:
                        # Steps that use per-regime processing (analyst models)
                        step_success = await self._execute_per_regime_step(
                            step_name, symbol, exchange, timeframe, data_dir, force_rerun
                        )
                    elif step_name in self.all_regime_steps:
                        # Steps that use all-regime processing (tactician models)
                        step_success = await self._execute_all_regime_step(
                            step_name, symbol, exchange, timeframe, data_dir, force_rerun
                        )
                    else:
                        # Standard steps (fallback)
                        step_success = await self._execute_standard_step(
                            step_name, symbol, exchange, timeframe, data_dir, force_rerun
                        )
                    
                    result.step_results[step_name] = step_success
                    
                    if step_success:
                        result.completed_steps += 1
                        self.logger.info(f'✅ Completed step: {step_name}')
                        
                        # Update continuity state
                        await self.continuity_manager.update_step_continuity(
                            symbol, exchange, timeframe, step_name, 
                            {'success': True, 'continuity_score': 1.0}
                        )
                    else:
                        result.failed_steps += 1
                        self.logger.error(f'❌ Failed step: {step_name}')
                        
                        # Update continuity state with failure
                        await self.continuity_manager.update_step_continuity(
                            symbol, exchange, timeframe, step_name, 
                            {'success': False, 'continuity_score': 0.0}
                        )
                        
                        if not self._should_continue_after_failure(step_name):
                            self.logger.error(f'🛑 Stopping pipeline due to critical step failure: {step_name}')
                            break
                    
                    # Validate continuity after each step based on regime type
                    should_validate = self.config.get('regime_continuity_validation', {}).get('validate_after_each_step', True)
                    
                    if should_validate:
                        # Determine validation type based on step
                        if step_name in self.regime_retagging_steps:
                            # Special validation for regime re-tagging steps
                            validation_result = await self._validate_regime_retagging_continuity(
                                step_name, symbol, exchange, timeframe, data_dir
                            )
                        elif step_name in self.per_regime_steps:
                            # Validation for HMM-retagged regime steps
                            validation_result = await self.continuity_validator.validate_step_continuity(
                                step_name, symbol, exchange, timeframe, data_dir
                            )
                        elif step_name in self.market_analysis_regime_steps:
                            # Validation for MARKET_ANALYSIS regime steps
                            validation_result = await self._validate_market_analysis_regime_continuity(
                                step_name, symbol, exchange, timeframe, data_dir
                            )
                        else:
                            # Standard validation
                            validation_result = await self.continuity_validator.validate_step_continuity(
                                step_name, symbol, exchange, timeframe, data_dir
                            )
                        
                        if not validation_result.get('is_valid', True):
                            self.logger.warning(f'⚠️ Continuity validation failed for {step_name}: {validation_result.get("issues", [])}')
                        
                        # Update continuity validation score
                        validation_score = validation_result.get('validation_score', 1.0)
                        if result.completed_steps > 0:
                            result.continuity_validation_score = (
                                result.continuity_validation_score * (result.completed_steps - 1) + validation_score
                            ) / result.completed_steps
                        else:
                            result.continuity_validation_score = validation_score
                            
                except Exception as e:
                    self.logger.exception(f'❌ Error executing step {step_name}: {e}')
                    result.step_results[step_name] = False
                    result.failed_steps += 1
                    
                    # Update continuity state with error
                    await self.continuity_manager.update_step_continuity(
                        symbol, exchange, timeframe, step_name, 
                        {'success': False, 'continuity_score': 0.0, 'error': str(e)}
                    )
                    
                    if not self._should_continue_after_failure(step_name):
                        break
            
            # Perform final pipeline continuity validation
            self.logger.info('🔍 Performing final pipeline continuity validation')
            final_validation = await self.continuity_validator.validate_pipeline_continuity(
                symbol, exchange, timeframe, data_dir, steps_to_run
            )
            result.continuity_validation_score = final_validation.get('overall_score', 0.0)
            
            # Determine overall success
            result.overall_success = (
                result.failed_steps == 0 and 
                result.continuity_validation_score >= 0.8
            )
            result.execution_end = datetime.now()
            
            # Save execution results
            await self._save_execution_results(result, symbol, exchange, timeframe, data_dir)
            
            if result.overall_success:
                self.logger.info(f'✅ Per-regime pipeline completed successfully for {exchange}_{symbol}_{timeframe}')
            else:
                self.logger.error(f'❌ Per-regime pipeline failed for {exchange}_{symbol}_{timeframe}')
                
            return result
            
        except Exception as e:
            self.logger.exception(f'❌ Error in per-regime pipeline execution: {e}')
            result.error_message = str(e)
            result.execution_end = datetime.now()
            return result

    @log_all_calls
    async def _execute_per_regime_step(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        force_rerun: bool
    ) -> bool:
        """Execute a step with per-regime processing (using HMM-retagged regimes).
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f'🔄 Executing per-regime step (HMM-retagged): {step_name}')
            
            # Execute using the final sub-pipeline structure
            from .sub_pipeline_final import ModelTrainingSubPipelineFinal, SubPipelineConfig
            
            # Create sub-pipeline configuration
            sub_config = SubPipelineConfig(
                mode=self.config.get('mode', 'full'),
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                custom_params=self.config
            )
            
            # Execute the specific sub-pipeline
            sub_pipeline = ModelTrainingSubPipelineFinal(sub_config)
            result = await sub_pipeline.execute_sub_pipeline(step_name, sub_config)
            
            # Check if step completed successfully
            if result.status.value == 'completed':
                self.logger.info(f'✅ Per-regime step {step_name} completed using HMM-retagged regimes')
                return True
            else:
                self.logger.error(f'❌ Per-regime step {step_name} failed: {result.error_message}')
                return False
                
        except Exception as e:
            self.logger.exception(f'❌ Error executing per-regime step {step_name}: {e}')
            return False

    @log_all_calls
    async def _execute_regime_retagging_step(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        force_rerun: bool
    ) -> bool:
        """Execute a step that handles regime re-tagging (HMM training).
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f'🔄 Executing regime re-tagging step: {step_name}')
            
            if step_name == 'hmm_training':
                # Execute HMM training which includes regime re-tagging
                from .sub_pipeline import ModelTrainingSubPipeline, SubPipelineConfig
                
                # Create sub-pipeline configuration
                sub_config = SubPipelineConfig(
                    mode=self.config.get('mode', 'full'),
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    force_rerun=force_rerun,
                    custom_params=self.config
                )
                
                # Execute HMM training sub-pipeline
                sub_pipeline = ModelTrainingSubPipeline(sub_config)
                result = await sub_pipeline.execute_sub_pipeline('hmm_training', sub_config)
                
                # Check if HMM training completed successfully
                if result.status.value == 'completed':
                    self.logger.info('✅ HMM training completed - regimes have been re-tagged')
                    return True
                else:
                    self.logger.error(f'❌ HMM training failed: {result.error_message}')
                    return False
            else:
                self.logger.error(f'❌ Unknown regime re-tagging step: {step_name}')
                return False
                
        except Exception as e:
            self.logger.exception(f'❌ Error executing regime re-tagging step {step_name}: {e}')
            return False

    @log_all_calls
    async def _execute_all_regime_step(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        force_rerun: bool
    ) -> bool:
        """Execute a step with all-regime processing (tactician models).
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f'🔄 Executing all-regime step (tactician models): {step_name}')
            
            # Execute using the final sub-pipeline structure
            from .sub_pipeline_final import ModelTrainingSubPipelineFinal, SubPipelineConfig
            
            # Create sub-pipeline configuration
            sub_config = SubPipelineConfig(
                mode=self.config.get('mode', 'full'),
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                custom_params=self.config
            )
            
            # Execute the specific sub-pipeline
            sub_pipeline = ModelTrainingSubPipelineFinal(sub_config)
            result = await sub_pipeline.execute_sub_pipeline(step_name, sub_config)
            
            # Check if step completed successfully
            if result.status.value == 'completed':
                self.logger.info(f'✅ All-regime step {step_name} completed successfully')
                return True
            else:
                self.logger.error(f'❌ All-regime step {step_name} failed: {result.error_message}')
                return False
                
        except Exception as e:
            self.logger.error(f'❌ Error executing all-regime step {step_name}: {e}')
            return False

    @log_all_calls
    async def _execute_market_analysis_regime_step(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        force_rerun: bool
    ) -> bool:
        """Execute a step that uses original MARKET_ANALYSIS regimes.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f'🔄 Executing MARKET_ANALYSIS regime step: {step_name}')
            
            if step_name == 'general_model_training':
                # Execute general model training using original MARKET_ANALYSIS regimes
                from .sub_pipeline import ModelTrainingSubPipeline, SubPipelineConfig
                
                # Create sub-pipeline configuration
                sub_config = SubPipelineConfig(
                    mode=self.config.get('mode', 'full'),
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    force_rerun=force_rerun,
                    custom_params=self.config
                )
                
                # Execute general model training sub-pipeline
                sub_pipeline = ModelTrainingSubPipeline(sub_config)
                result = await sub_pipeline.execute_sub_pipeline('general_model_training', sub_config)
                
                # Check if general model training completed successfully
                if result.status.value == 'completed':
                    self.logger.info('✅ General model training completed using MARKET_ANALYSIS regimes')
                    return True
                else:
                    self.logger.error(f'❌ General model training failed: {result.error_message}')
                    return False
            else:
                self.logger.error(f'❌ Unknown MARKET_ANALYSIS regime step: {step_name}')
                return False
                
        except Exception as e:
            self.logger.exception(f'❌ Error executing MARKET_ANALYSIS regime step {step_name}: {e}')
            return False

    @log_all_calls
    async def _execute_standard_step(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        force_rerun: bool
    ) -> bool:
        """Execute a step with standard processing.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f'🔄 Executing standard step: {step_name}')
            
            # Get step function from pipeline integrator
            step_function = await self.pipeline_integrator.get_step_function(step_name)
            if step_function is None:
                self.logger.error(f'❌ No step function found for {step_name}')
                return False
            
            # Execute the step function
            result = await step_function(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config
            )
            
            # Handle different result types
            if isinstance(result, bool):
                return result
            elif isinstance(result, dict):
                return result.get('success', False)
            else:
                self.logger.warning(f'⚠️ Unexpected result type for {step_name}: {type(result)}')
                return False
                
        except Exception as e:
            self.logger.exception(f'❌ Error executing standard step {step_name}: {e}')
            return False
    @log_all_calls
    def _should_continue_after_failure(self, step_name: str) -> bool:
        """Determine if pipeline should continue after a step failure.
        
        Args:
            step_name: Name of the failed step
            
        Returns:
            True if pipeline should continue
        """
        critical_steps = [
            'step04_regime_data_splitting', 'step05_labeling', 
            'step06_feature_engineering', 'step21_saving'
        ]
        return step_name not in critical_steps

    @log_all_calls
    async def _save_execution_results(
        self, 
        result: PipelineExecutionResult, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> None:
        """Save pipeline execution results.
        
        Args:
            result: Pipeline execution result
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            training_dir = Path(data_dir) / 'training'
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert result to dictionary
            result_dict = asdict(result)
            result_dict['execution_start'] = result.execution_start.isoformat()
            result_dict['execution_end'] = result.execution_end.isoformat() if result.execution_end else None
            
            # Save execution results
            execution_file = training_dir / f'{exchange}_{symbol}_{timeframe}_pipeline_execution_results.json'
            safe_json_dump(result_dict, execution_file)
            self.logger.info(f'✅ Saved pipeline execution results: {execution_file}')
            
            # Save continuity validation results
            validation_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_continuity_validation.json'
            validation_data = {
                'continuity_score': result.continuity_validation_score,
                'overall_success': result.overall_success,
                'timestamp': datetime.now().isoformat(),
                'step_results': result.step_results
            }
            safe_json_dump(validation_data, validation_file)
            
        except Exception as e:
            self.logger.error(f'❌ Error saving execution results: {e}')

    @traced(span_name='get_pipeline_status')
    async def get_pipeline_status(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Dict[str, Any]:
        """Get the current status of the pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Pipeline status information
        """
        try:
            # Get continuity report
            continuity_report = await self.continuity_manager.get_continuity_report(
                symbol, exchange, timeframe
            )
            
            # Load validation results
            training_dir = Path(data_dir) / 'training'
            validation_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_continuity_validation.json'
            validation_results = {}
            if validation_file.exists():
                validation_results = safe_json_load(validation_file)
            
            # Load execution results
            execution_file = training_dir / f'{exchange}_{symbol}_{timeframe}_pipeline_execution_results.json'
            execution_results = {}
            if execution_file.exists():
                execution_results = safe_json_load(execution_file)
            
            # Calculate pipeline health
            pipeline_health = self._calculate_pipeline_health(
                continuity_report, validation_results, execution_results
            )
            
            status = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'continuity_report': continuity_report,
                'validation_results': validation_results,
                'execution_results': execution_results,
                'pipeline_health': pipeline_health
            }
            
            return status
            
        except Exception as e:
            self.logger.exception(f'❌ Error getting pipeline status: {e}')
            return {
                'error': str(e), 
                'timestamp': datetime.now().isoformat()
            }
    @log_all_calls
    def _calculate_pipeline_health(
        self, 
        continuity_report: Dict[str, Any], 
        validation_results: Dict[str, Any], 
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall pipeline health.
        
        Args:
            continuity_report: Continuity report
            validation_results: Validation results
            execution_results: Execution results
            
        Returns:
            Pipeline health information
        """
        try:
            health_score = 1.0
            issues = []
            warnings = []
            
            # Check continuity report
            if continuity_report and not continuity_report.get('error'):
                continuity_score = continuity_report.get('continuity_score', 1.0)
                if continuity_score < 0.8:
                    health_score *= continuity_score
                    issues.append(f'Low continuity score: {continuity_score:.2f}')
                
                total_steps = continuity_report.get('total_steps', 0)
                completed_steps = continuity_report.get('completed_steps', 0)
                if total_steps > 0 and completed_steps < total_steps:
                    completion_ratio = completed_steps / total_steps
                    health_score *= completion_ratio
                    warnings.append(f'Incomplete pipeline: {completed_steps}/{total_steps} steps completed')
            elif continuity_report.get('error'):
                health_score *= 0.3
                issues.append(f'Continuity error: {continuity_report["error"]}')
            
            # Check validation results
            if validation_results:
                continuity_score = validation_results.get('continuity_score', 0.0)
                if continuity_score < 0.8:
                    health_score *= continuity_score
                    issues.append(f'Low validation score: {continuity_score:.2f}')
            
            # Check execution results
            if execution_results:
                overall_success = execution_results.get('overall_success', False)
                if not overall_success:
                    health_score *= 0.3
                    issues.append('Pipeline execution failed')
                
                failed_steps = execution_results.get('failed_steps', 0)
                if failed_steps > 0:
                    health_score *= 0.8
                    warnings.append(f'{failed_steps} steps failed')
            
            # Determine health status
            if health_score >= 0.9:
                status = 'excellent'
            elif health_score >= 0.8:
                status = 'good'
            elif health_score >= 0.6:
                status = 'fair'
            elif health_score >= 0.4:
                status = 'poor'
            else:
                status = 'critical'
            
            return {
                'health_score': health_score,
                'status': status,
                'issues': issues,
                'warnings': warnings,
                'recommendations': self._generate_health_recommendations(health_score, issues, warnings)
            }
            
        except Exception as e:
            self.logger.error(f'❌ Error calculating pipeline health: {e}')
            return {
                'health_score': 0.0,
                'status': 'unknown',
                'issues': [f'Health calculation error: {str(e)}'],
                'warnings': [],
                'recommendations': ['Fix health calculation error']
            }

    @log_all_calls
    def _generate_health_recommendations(
        self, 
        health_score: float, 
        issues: List[str], 
        warnings: List[str]
    ) -> List[str]:
        """Generate health recommendations.
        
        Args:
            health_score: Overall health score
            issues: List of issues
            warnings: List of warnings
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if health_score < 0.8:
            recommendations.append('Overall pipeline health is below optimal. Review and address all issues.')
        
        if any('regime' in issue.lower() for issue in issues):
            recommendations.append('Address regime-related issues to improve pipeline health.')
        
        if any('validation' in issue.lower() for issue in issues):
            recommendations.append('Improve validation processes to ensure data quality.')
        
        if any('execution' in issue.lower() for issue in issues):
            recommendations.append('Review and fix execution failures.')
        
        if any('continuity' in issue.lower() for issue in issues):
            recommendations.append('Check regime continuity management and data flow.')
        
        if not recommendations:
            recommendations.append('Pipeline health is good. Continue monitoring.')
        
        return recommendations


# Global orchestrator instance
per_regime_pipeline_orchestrator = PerRegimePipelineOrchestrator()


@traced(span_name='run_per_regime_pipeline')
@validates()
@handles_errors
async def run_per_regime_pipeline(
    symbol: str, 
    exchange: str, 
    timeframe: str, 
    data_dir: Optional[str] = None, 
    force_rerun: bool = False, 
    steps_to_run: Optional[List[str]] = None, 
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the complete per-regime pipeline.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun all steps
        steps_to_run: Specific steps to run
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info('🚀 Starting Per-Regime Pipeline Orchestration')
    
    if config is None:
        config = {}
    
    if data_dir is None:
        data_dir = standardized_parquet_handler.get_standardized_path('processed_data', exchange, symbol)
    
    orchestrator = PerRegimePipelineOrchestrator(config)
    result = await orchestrator.execute_per_regime_pipeline(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun,
        steps_to_run=steps_to_run
    )
    
    if result.overall_success:
        logger.info('✅ Per-Regime Pipeline completed successfully')
    else:
        logger.error(f'❌ Per-Regime Pipeline failed: {result.error_message}')
    
    return result.overall_success


if __name__ == '__main__':
    async def test() -> None:
        """Test the per-regime pipeline."""
        success = await run_per_regime_pipeline(
            symbol='ETHUSDT', 
            exchange='BINANCE', 
            timeframe='1m', 
            data_dir='data_cache'
        )
        tprint(f'Per-regime pipeline result: {success}')
    
    asyncio.run(test())
