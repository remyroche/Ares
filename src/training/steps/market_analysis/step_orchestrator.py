#!/usr/bin/env python3
"""
Market Analysis Step Orchestrator

This module provides a comprehensive step orchestrator that ensures:
1. Proper flow between pipeline steps
2. Step dependency validation
3. Step execution monitoring
4. Error handling and recovery
5. Progress tracking and reporting
6. Data consistency across steps
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
)
from src.utils.enhanced_common_operations import (
    data_access_manager,
    data_analysis_manager,
    performance_monitor,
    monitor_async_operation,
)
from src.utils.validator_orchestrator import ValidatorOrchestrator
from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import comprehensive_protection
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.security_framework import SecurityFramework

logger = system_logger.getChild("MarketAnalysisStepOrchestrator")


class StepDependency:
    """Represents a step dependency with validation rules."""
    
    def __init__(self, step_name: str, required_outputs: List[str], validation_rules: Dict[str, Any]):
        self.step_name = step_name
        self.required_outputs = required_outputs
        self.validation_rules = validation_rules
        self.satisfied = False
        self.satisfied_at = None
    
    def check_satisfaction(self, pipeline_state: Dict[str, Any]) -> bool:
        """Check if this dependency is satisfied by the current pipeline state."""
        if self.step_name not in pipeline_state:
            return False
        
        step_result = pipeline_state[self.step_name]
        if not step_result.get('success', False):
            return False
        
        # Check if all required outputs are present
        for output in self.required_outputs:
            if output not in step_result:
                return False
        
        # Apply validation rules
        for rule_name, rule_config in self.validation_rules.items():
            if not self._apply_validation_rule(rule_name, rule_config, step_result):
                return False
        
        self.satisfied = True
        self.satisfied_at = get_current_datetime().isoformat()
        return True
    
    def _apply_validation_rule(self, rule_name: str, rule_config: Dict[str, Any], step_result: Dict[str, Any]) -> bool:
        """Apply a specific validation rule."""
        try:
            if rule_name == "file_exists":
                file_path = step_result.get(rule_config.get('output_key'))
                return safe_file_exists(file_path) if file_path else False
            
            elif rule_name == "data_quality":
                data = step_result.get(rule_config.get('output_key'))
                if data is None:
                    return False
                
                # Basic data quality checks
                if hasattr(data, 'shape'):  # DataFrame
                    return data.shape[0] > 0 and data.shape[1] > 0
                elif isinstance(data, dict):
                    return len(data) > 0
                elif isinstance(data, list):
                    return len(data) > 0
                
                return True
            
            elif rule_name == "custom":
                # Custom validation function
                validation_func = rule_config.get('function')
                if validation_func and callable(validation_func):
                    return validation_func(step_result)
                
                return True
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Validation rule {rule_name} failed: {e}")
            return False


class StepExecutionResult:
    """Represents the result of a step execution."""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.success = False
        self.error = None
        self.outputs = {}
        self.warnings = []
        self.metadata = {}
    
    def start_execution(self):
        """Mark the start of step execution."""
        self.start_time = time.time()
    
    def end_execution(self, success: bool, error: str = None, outputs: Dict[str, Any] = None, warnings: List[str] = None):
        """Mark the end of step execution."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error
        self.outputs = outputs or {}
        self.warnings = warnings or []
        self.metadata['timestamp'] = get_current_datetime().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'step_name': self.step_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'success': self.success,
            'error': self.error,
            'outputs': self.outputs,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


class MarketAnalysisStepOrchestrator:
    """Orchestrator for market analysis pipeline steps with comprehensive flow control."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('MarketAnalysisStepOrchestrator')
        self.validator_orchestrator = ValidatorOrchestrator()
        self.data_quality = DataQualityFramework()
        self.security = SecurityFramework()
        
        # Step definitions and dependencies
        self.step_definitions = self._define_steps()
        self.step_dependencies = self._define_dependencies()
        self.step_execution_order = self._calculate_execution_order()
        
        # Execution state
        self.pipeline_state = {}
        self.execution_results = {}
        self.current_step = None
        self.execution_start_time = None
        self.total_execution_time = None
        
        # Progress tracking
        self.progress_callback = None
        self.progress_update_interval = 5.0  # seconds
        self.last_progress_update = 0
    
    def _define_steps(self) -> Dict[str, Dict[str, Any]]:
        """Define all available steps in the market analysis pipeline."""
        return {
            'data_collection': {
                'name': 'Data Collection',
                'description': 'Collect and consolidate market data',
                'module': 'src.training.steps.market_analysis.enhanced_market_analysis_pipeline',
                'class': 'DataCollectionStep',
                'estimated_duration': 300,  # 5 minutes
                'critical': True,
                'retry_count': 3,
                'timeout': 600  # 10 minutes
            },
            'hmm_clustering': {
                'name': 'HMM Clustering',
                'description': 'Perform HMM regime discovery and clustering',
                'module': 'src.training.steps.market_analysis.enhanced_market_analysis_pipeline',
                'class': 'HMMClusteringStep',
                'estimated_duration': 600,  # 10 minutes
                'critical': True,
                'retry_count': 2,
                'timeout': 1200  # 20 minutes
            },
            'feature_engineering': {
                'name': 'Feature Engineering',
                'description': 'Engineer features for machine learning',
                'module': 'src.training.steps.market_analysis.enhanced_market_analysis_pipeline',
                'class': 'FeatureEngineeringStep',
                'estimated_duration': 900,  # 15 minutes
                'critical': True,
                'retry_count': 2,
                'timeout': 1800  # 30 minutes
            }
        }
    
    def _define_dependencies(self) -> Dict[str, List[StepDependency]]:
        """Define step dependencies."""
        return {
            'data_collection': [],  # No dependencies
            'hmm_clustering': [
                StepDependency(
                    step_name='data_collection',
                    required_outputs=['data_file', 'data_exists'],
                    validation_rules={
                        'file_exists': {'output_key': 'data_file'},
                        'data_quality': {'output_key': 'data_file'}
                    }
                )
            ],
            'feature_engineering': [
                StepDependency(
                    step_name='hmm_clustering',
                    required_outputs=['regime_model', 'regime_labels'],
                    validation_rules={
                        'data_quality': {'output_key': 'regime_model'}
                    }
                )
            ]
        }
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate the execution order based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name: str):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving step: {step_name}")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            
            # Visit dependencies first
            for dependency in self.step_dependencies.get(step_name, []):
                visit(dependency.step_name)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        # Visit all steps
        for step_name in self.step_definitions.keys():
            if step_name not in visited:
                visit(step_name)
        
        return order
    
    @comprehensive_protection(
        operation_name="orchestrator_initialization",
        operation_type="orchestrator_initialization",
        context="orchestrator_initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the step orchestrator."""
        start_time = time.time()
        self.logger.info("🚀 Initializing Market Analysis Step Orchestrator...")
        
        try:
            # Initialize frameworks
            await self.data_quality.initialize()
            await self.security.initialize()
            await data_access_manager.initialize()
            await data_analysis_manager.initialize()
            
            # Validate step definitions
            validation_result = await self._validate_step_definitions()
            if not validation_result.get('valid', False):
                self.logger.error(f"❌ Step definitions validation failed: {validation_result.get('error')}")
                return False
            
            # Validate dependencies
            dependency_validation = await self._validate_dependencies()
            if not dependency_validation.get('valid', False):
                self.logger.error(f"❌ Dependencies validation failed: {dependency_validation.get('error')}")
                return False
            
            duration = time.time() - start_time
            self.logger.info(f"✅ Market Analysis Step Orchestrator initialized successfully in {duration:.3f}s")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Failed to initialize orchestrator: {e}")
            return False
    
    @comprehensive_protection(
        operation_name="orchestrator_execution",
        operation_type="orchestrator_execution",
        context="orchestrator_execution"
    )
    async def execute_pipeline(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete market analysis pipeline."""
        start_time = time.time()
        self.execution_start_time = start_time
        self.logger.info("🎯 Starting Market Analysis Pipeline Execution...")
        
        try:
            # Validate training input
            input_validation = await self._validate_training_input(training_input)
            if not input_validation.get('valid', False):
                return {
                    'success': False,
                    'error': f"Training input validation failed: {input_validation.get('error')}",
                    'timestamp': get_current_datetime().isoformat()
                }
            
            # Execute steps in order
            for step_name in self.step_execution_order:
                self.current_step = step_name
                
                # Check if step should be executed
                if not await self._should_execute_step(step_name, training_input):
                    self.logger.info(f"⏭️ Skipping step {step_name} (conditions not met)")
                    continue
                
                # Execute step
                step_result = await self._execute_step(step_name, training_input)
                self.execution_results[step_name] = step_result
                
                # Update pipeline state
                self.pipeline_state[step_name] = step_result.to_dict()
                
                # Check if step failed
                if not step_result.success:
                    self.logger.error(f"❌ Step {step_name} failed: {step_result.error}")
                    return {
                        'success': False,
                        'error': f"Pipeline failed at step {step_name}: {step_result.error}",
                        'failed_step': step_name,
                        'execution_results': {k: v.to_dict() for k, v in self.execution_results.items()},
                        'timestamp': get_current_datetime().isoformat()
                    }
                
                # Update progress
                await self._update_progress(step_name, len(self.step_execution_order))
            
            # Final validation
            final_validation = await self._validate_pipeline_completion()
            if not final_validation.get('valid', False):
                self.logger.warning(f"⚠️ Final validation failed: {final_validation.get('warnings')}")
            
            self.total_execution_time = time.time() - start_time
            self.logger.info("🎉 Market Analysis Pipeline completed successfully!")
            
            return {
                'success': True,
                'execution_results': {k: v.to_dict() for k, v in self.execution_results.items()},
                'pipeline_state': self.pipeline_state,
                'total_execution_time': self.total_execution_time,
                'timestamp': get_current_datetime().isoformat()
            }
            
        except Exception as e:
            self.total_execution_time = time.time() - start_time
            self.logger.exception(f"❌ Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_results': {k: v.to_dict() for k, v in self.execution_results.items()},
                'total_execution_time': self.total_execution_time,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _validate_step_definitions(self) -> Dict[str, Any]:
        """Validate step definitions."""
        try:
            for step_name, step_def in self.step_definitions.items():
                # Check required fields
                required_fields = ['name', 'description', 'module', 'class']
                missing_fields = [field for field in required_fields if field not in step_def]
                
                if missing_fields:
                    return {
                        'valid': False,
                        'error': f"Step {step_name} missing required fields: {missing_fields}"
                    }
                
                # Check if module and class exist
                try:
                    module = __import__(step_def['module'], fromlist=[step_def['class']])
                    if not hasattr(module, step_def['class']):
                        return {
                            'valid': False,
                            'error': f"Class {step_def['class']} not found in module {step_def['module']}"
                        }
                except ImportError as e:
                    return {
                        'valid': False,
                        'error': f"Cannot import module {step_def['module']}: {e}"
                    }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate step dependencies."""
        try:
            for step_name, dependencies in self.step_dependencies.items():
                for dependency in dependencies:
                    if dependency.step_name not in self.step_definitions:
                        return {
                            'valid': False,
                            'error': f"Step {step_name} depends on undefined step: {dependency.step_name}"
                        }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_training_input(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training input parameters."""
        try:
            required_keys = ['symbol', 'exchange', 'timeframe', 'data_dir']
            missing_keys = [key for key in required_keys if key not in training_input]
            
            if missing_keys:
                return {
                    'valid': False,
                    'error': f"Missing required training input keys: {missing_keys}"
                }
            
            # Validate symbol format
            symbol = training_input.get('symbol', '')
            if not symbol or not isinstance(symbol, str):
                return {
                    'valid': False,
                    'error': 'Symbol must be a non-empty string'
                }
            
            # Validate exchange
            exchange = training_input.get('exchange', '')
            valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
            if exchange not in valid_exchanges:
                return {
                    'valid': False,
                    'error': f'Exchange must be one of: {valid_exchanges}'
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _should_execute_step(self, step_name: str, training_input: Dict[str, Any]) -> bool:
        """Determine if a step should be executed based on conditions."""
        try:
            step_def = self.step_definitions[step_name]
            
            # Check if step is enabled in config
            if not self.config.get(f'enable_{step_name}', True):
                return False
            
            # Check dependencies
            dependencies = self.step_dependencies.get(step_name, [])
            for dependency in dependencies:
                if not dependency.check_satisfaction(self.pipeline_state):
                    self.logger.warning(f"⚠️ Step {step_name} dependencies not satisfied")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error checking step execution conditions: {e}")
            return False
    
    async def _execute_step(self, step_name: str, training_input: Dict[str, Any]) -> StepExecutionResult:
        """Execute a single step with comprehensive monitoring and error handling."""
        result = StepExecutionResult(step_name)
        result.start_execution()
        
        try:
            step_def = self.step_definitions[step_name]
            self.logger.info(f"🔄 Executing step: {step_name} - {step_def['name']}")
            
            # Import and instantiate step class
            module = __import__(step_def['module'], fromlist=[step_def['class']])
            step_class = getattr(module, step_def['class'])
            step_instance = step_class(self.config)
            
            # Initialize step
            init_success = await step_instance.initialize()
            if not init_success:
                result.end_execution(False, "Failed to initialize step")
                return result
            
            # Execute step with monitoring
            async with monitor_async_operation(f"step_{step_name}"):
                step_output = await step_instance.execute(training_input, self.pipeline_state)
            
            # Validate step output
            output_validation = await self._validate_step_output(step_name, step_output)
            if not output_validation.get('valid', False):
                result.end_execution(False, f"Step output validation failed: {output_validation.get('error')}")
                return result
            
            # Record successful execution
            result.end_execution(
                success=True,
                outputs=step_output,
                warnings=output_validation.get('warnings', [])
            )
            
            self.logger.info(f"✅ Step {step_name} completed successfully in {result.duration:.3f}s")
            return result
            
        except Exception as e:
            result.end_execution(False, str(e))
            self.logger.exception(f"❌ Step {step_name} failed: {e}")
            return result
    
    async def _validate_step_output(self, step_name: str, step_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step output."""
        try:
            warnings = []
            
            # Check if output is a dictionary
            if not isinstance(step_output, dict):
                return {
                    'valid': False,
                    'error': 'Step output must be a dictionary'
                }
            
            # Check for success flag
            if not step_output.get('success', False):
                return {
                    'valid': False,
                    'error': 'Step output indicates failure'
                }
            
            # Step-specific validation
            if step_name == 'data_collection':
                if not step_output.get('data_exists', False):
                    warnings.append("Data collection did not produce expected data")
            
            elif step_name == 'hmm_clustering':
                if not step_output.get('regime_model'):
                    warnings.append("HMM clustering did not produce regime model")
            
            elif step_name == 'feature_engineering':
                if not step_output.get('features'):
                    warnings.append("Feature engineering did not produce features")
            
            return {
                'valid': True,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    async def _validate_pipeline_completion(self) -> Dict[str, Any]:
        """Validate that the pipeline completed successfully."""
        try:
            warnings = []
            
            # Check that all critical steps completed
            critical_steps = [name for name, def_ in self.step_definitions.items() if def_.get('critical', False)]
            for step_name in critical_steps:
                if step_name not in self.execution_results:
                    warnings.append(f"Critical step {step_name} was not executed")
                elif not self.execution_results[step_name].success:
                    warnings.append(f"Critical step {step_name} failed")
            
            # Check execution time
            if self.total_execution_time and self.total_execution_time > 3600:  # 1 hour
                warnings.append(f"Pipeline execution took {self.total_execution_time:.1f}s, which is longer than expected")
            
            return {
                'valid': len(warnings) == 0,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    async def _update_progress(self, completed_step: str, total_steps: int) -> None:
        """Update progress and call progress callback if available."""
        try:
            current_time = time.time()
            
            # Only update progress if enough time has passed
            if current_time - self.last_progress_update < self.progress_update_interval:
                return
            
            progress_percentage = (len(self.execution_results) / total_steps) * 100
            
            if self.progress_callback:
                await self.progress_callback(
                    completed_step=completed_step,
                    progress_percentage=progress_percentage,
                    execution_results=self.execution_results,
                    pipeline_state=self.pipeline_state
                )
            
            self.last_progress_update = current_time
            
        except Exception as e:
            self.logger.warning(f"⚠️ Progress update failed: {e}")
    
    def set_progress_callback(self, callback: Callable):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        return {
            'orchestrator_name': 'Market Analysis Step Orchestrator',
            'total_steps': len(self.step_definitions),
            'executed_steps': len(self.execution_results),
            'successful_steps': len([r for r in self.execution_results.values() if r.success]),
            'failed_steps': len([r for r in self.execution_results.values() if not r.success]),
            'total_execution_time': self.total_execution_time,
            'current_step': self.current_step,
            'execution_results': {k: v.to_dict() for k, v in self.execution_results.items()},
            'pipeline_state': self.pipeline_state,
            'timestamp': get_current_datetime().isoformat()
        }


# Main orchestrator execution function
async def run_market_analysis_orchestrator(
    symbol: str,
    exchange: str,
    timeframe: str = '1m',
    data_dir: str = 'data_cache',
    **config
) -> Dict[str, Any]:
    """Run the market analysis pipeline using the step orchestrator."""
    
    # Prepare training input
    training_input = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'timestamp': get_current_datetime().isoformat()
    }
    
    # Initialize orchestrator
    orchestrator = MarketAnalysisStepOrchestrator(config)
    
    # Initialize orchestrator
    init_success = await orchestrator.initialize()
    if not init_success:
        return {
            'success': False,
            'error': 'Failed to initialize orchestrator',
            'timestamp': get_current_datetime().isoformat()
        }
    
    # Execute pipeline
    result = await orchestrator.execute_pipeline(training_input)
    
    # Add orchestrator summary
    if result.get('success', False):
        result['orchestrator_summary'] = orchestrator.get_execution_summary()
    
    return result


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'enable_data_collection': True,
            'enable_hmm_clustering': True,
            'enable_feature_engineering': True,
            'force_rerun': True,
            'random_state': 42,
        }
        
        result = await run_market_analysis_orchestrator(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache',
            **config
        )
        
        print(f"Orchestrator result: {result}")
    
    asyncio.run(main())