#!/usr/bin/env python3
"""
Enhanced Market Analysis Pipeline

This module provides a comprehensive market analysis pipeline with:
1. Step-by-step validation at each stage
2. Comprehensive decorators for data protection and error handling
3. Common utilities for data formatting, analysis, and access
4. Pipeline orchestration with proper flow control
5. Data protection mechanisms for all operations
6. Comprehensive validation framework for pipeline integrity
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
from src.utils.data_formatting_framework import DataFormattingFramework, DataFormat
from src.utils.validator_orchestrator import ValidatorOrchestrator
from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import handles_errors, data_protection, operation_monitoring
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.security_framework import SecurityFramework

logger = system_logger.getChild("EnhancedMarketAnalysisPipeline")

class MarketAnalysisPipelineStep:
    """Base class for market analysis pipeline steps with comprehensive validation and protection."""
    
    def __init__(self, step_name: str, config: Dict[str, Any]):
        self.step_name = step_name
        self.config = config
        self.logger = system_logger.getChild(f"MarketAnalysisStep_{step_name}")
        self.validator_orchestrator = ValidatorOrchestrator()
        self.data_formatter = DataFormattingFramework()
        self.data_quality = DataQualityFramework()
        self.security = SecurityFramework()
        self.step_timings = {}
        self.validation_results = {}
        
    @handles_errors(Exception, fallback=None, context="step_initialization")
    @data_protection(operation_type="initialization")
    @operation_monitoring(operation_name="step_initialization")
    async def initialize(self) -> bool:
        """Initialize the pipeline step with comprehensive validation."""
        start_time = time.time()
        self.logger.info(f"🚀 Initializing {self.step_name}...")
        
        try:
            # Validate configuration
            config_validation = await self._validate_configuration()
            if not config_validation.get('valid', False):
                self.logger.error(f"❌ Configuration validation failed: {config_validation.get('error')}")
                return False
            
            # Initialize data quality framework
            await self.data_quality.initialize()
            
            # Initialize security framework
            await self.security.initialize()
            
            # Log initialization success
            duration = time.time() - start_time
            self.step_timings['initialization'] = duration
            self.logger.info(f"✅ {self.step_name} initialized successfully in {duration:.3f}s")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize {self.step_name}: {e}")
            return False
    
    @handles_errors(Exception, fallback=None, context="step_execution")
    @data_protection(operation_type="execution")
    @operation_monitoring(operation_name="step_execution")
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline step with comprehensive validation and protection."""
        start_time = time.time()
        self.logger.info(f"🎯 Executing {self.step_name}...")
        
        try:
            # Pre-execution validation
            pre_validation = await self._pre_execution_validation(training_input, pipeline_state)
            if not pre_validation.get('passed', False):
                return {
                    'success': False,
                    'error': f"Pre-execution validation failed: {pre_validation.get('error')}",
                    'step_name': self.step_name,
                    'timestamp': get_current_datetime().isoformat()
                }
            
            # Execute the main step logic
            result = await self._execute_main_logic(training_input, pipeline_state)
            
            # Post-execution validation
            post_validation = await self._post_execution_validation(result, training_input, pipeline_state)
            if not post_validation.get('passed', False):
                self.logger.warning(f"⚠️ Post-execution validation failed: {post_validation.get('error')}")
                result['warnings'] = result.get('warnings', []) + [post_validation.get('error')]
            
            # Log execution success
            duration = time.time() - start_time
            self.step_timings['execution'] = duration
            result['execution_time'] = duration
            result['step_name'] = self.step_name
            result['timestamp'] = get_current_datetime().isoformat()
            
            self.logger.info(f"✅ {self.step_name} executed successfully in {duration:.3f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Failed to execute {self.step_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': self.step_name,
                'execution_time': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate step configuration."""
        try:
            required_config_keys = self.get_required_config_keys()
            missing_keys = [key for key in required_config_keys if key not in self.config]
            
            if missing_keys:
                return {
                    'valid': False,
                    'error': f"Missing required configuration keys: {missing_keys}"
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _pre_execution_validation(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pre-execution validation."""
        try:
            # Validate training input
            if not isinstance(training_input, dict):
                return {'passed': False, 'error': 'training_input must be a dictionary'}
            
            # Validate pipeline state
            if not isinstance(pipeline_state, dict):
                return {'passed': False, 'error': 'pipeline_state must be a dictionary'}
            
            # Run step-specific pre-validation
            step_validation = await self._run_step_validator(
                f"{self.step_name}_pre_validation",
                training_input,
                pipeline_state,
                self.config
            )
            
            return {
                'passed': step_validation.get('validation_passed', False),
                'error': step_validation.get('error', 'Pre-validation failed')
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def _post_execution_validation(self, result: Dict[str, Any], training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform post-execution validation."""
        try:
            # Validate result structure
            if not isinstance(result, dict):
                return {'passed': False, 'error': 'Result must be a dictionary'}
            
            # Run step-specific post-validation
            step_validation = await self._run_step_validator(
                f"{self.step_name}_post_validation",
                result,
                pipeline_state,
                self.config
            )
            
            return {
                'passed': step_validation.get('validation_passed', False),
                'error': step_validation.get('error', 'Post-validation failed')
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def _run_step_validator(self, validator_name: str, data: Dict[str, Any], pipeline_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run step validator using the validator orchestrator."""
        try:
            return await self.validator_orchestrator.run_step_validator(
                validator_name,
                data,
                pipeline_state,
                config,
                validation_level='CRITICAL'
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Validator {validator_name} failed: {e}")
            return {'validation_passed': False, 'error': str(e)}
    
    def get_required_config_keys(self) -> List[str]:
        """Get list of required configuration keys for this step."""
        return ['symbol', 'exchange', 'timeframe', 'data_dir']
    
    async def _execute_main_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main logic for this step. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_main_logic")


class DataCollectionStep(MarketAnalysisPipelineStep):
    """Step 1: Data Collection with comprehensive validation and protection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("data_collection", config)
    
    @handles_errors(Exception, fallback=None, context="data_collection")
    @data_protection(operation_type="data_collection")
    @operation_monitoring(operation_name="data_collection")
    async def _execute_main_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection with comprehensive validation."""
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            self.logger.info(f"📊 Collecting data for {symbol} on {exchange}")
            
            # Ensure data directory exists
            ensure_directory(data_dir)
            
            # Check if data already exists
            data_file = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            if safe_file_exists(data_file):
                self.logger.info(f"✅ Data already exists: {data_file}")
                return {
                    'success': True,
                    'data_file': str(data_file),
                    'data_exists': True,
                    'message': 'Data collection completed - data already exists'
                }
            
            # Import and run data collection
            from src.training.steps.data_collection.step01_data_collection_main import run_data_collection
            
            # Execute data collection
            collection_result = await run_data_collection(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                config=self.config
            )
            
            if collection_result.get('success', False):
                self.logger.info("✅ Data collection completed successfully")
                return {
                    'success': True,
                    'data_file': collection_result.get('data_file'),
                    'data_exists': True,
                    'message': 'Data collection completed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': collection_result.get('error', 'Data collection failed'),
                    'message': 'Data collection failed'
                }
                
        except Exception as e:
            self.logger.exception(f"❌ Data collection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Data collection failed with exception'
            }


class HMMClusteringStep(MarketAnalysisPipelineStep):
    """Step 2: HMM Clustering with comprehensive validation and protection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("hmm_clustering", config)
    
    @handles_errors(Exception, fallback=None, context="hmm_clustering")
    @data_protection(operation_type="hmm_clustering")
    @operation_monitoring(operation_name="hmm_clustering")
    async def _execute_main_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HMM clustering with comprehensive validation."""
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            self.logger.info(f"🧠 Running HMM clustering for {symbol} on {exchange}")
            
            # Import and run HMM clustering
            from src.training.steps.hmm_clustering import run_enhanced_step
            
            # Execute HMM clustering
            clustering_result = await run_enhanced_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=self.config.get('force_rerun', False)
            )
            
            if clustering_result.get('success', False):
                self.logger.info("✅ HMM clustering completed successfully")
                return {
                    'success': True,
                    'regime_model': clustering_result.get('regime_model'),
                    'regime_labels': clustering_result.get('regime_labels'),
                    'message': 'HMM clustering completed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': clustering_result.get('error', 'HMM clustering failed'),
                    'message': 'HMM clustering failed'
                }
                
        except Exception as e:
            self.logger.exception(f"❌ HMM clustering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'HMM clustering failed with exception'
            }


class FeatureEngineeringStep(MarketAnalysisPipelineStep):
    """Step 3: Feature Engineering with comprehensive validation and protection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("feature_engineering", config)
    
    @handles_errors(Exception, fallback=None, context="feature_engineering")
    @data_protection(operation_type="feature_engineering")
    @operation_monitoring(operation_name="feature_engineering")
    async def _execute_main_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering with comprehensive validation."""
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            self.logger.info(f"🔧 Running feature engineering for {symbol} on {exchange}")
            
            # Import and run feature engineering
            from src.training.steps.market_analysis.step06_feature_engineering import FeatureEngineeringStep as BaseFeatureEngineeringStep
            
            # Execute feature engineering
            feature_engineer = BaseFeatureEngineeringStep()
            engineering_result = await feature_engineer.engineer_features(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            if engineering_result.get('success', False):
                self.logger.info("✅ Feature engineering completed successfully")
                return {
                    'success': True,
                    'features': engineering_result.get('features'),
                    'feature_count': engineering_result.get('feature_count'),
                    'message': 'Feature engineering completed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': engineering_result.get('error', 'Feature engineering failed'),
                    'message': 'Feature engineering failed'
                }
                
        except Exception as e:
            self.logger.exception(f"❌ Feature engineering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Feature engineering failed with exception'
            }


class EnhancedMarketAnalysisPipeline:
    """Enhanced Market Analysis Pipeline with comprehensive validation, decorators, and utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('EnhancedMarketAnalysisPipeline')
        self.data_formatter = DataFormattingFramework()
        self.validator_orchestrator = ValidatorOrchestrator()
        self.data_quality = DataQualityFramework()
        self.security = SecurityFramework()
        
        # Initialize pipeline steps
        self.steps = {
            'data_collection': DataCollectionStep(config),
            'hmm_clustering': HMMClusteringStep(config),
            'feature_engineering': FeatureEngineeringStep(config),
        }
        
        self.pipeline_state = {}
        self.execution_results = {}
        self.pipeline_timings = {}
    
    @handles_errors(Exception, fallback=False, context="pipeline_initialization")
    @data_protection(operation_type="pipeline_initialization")
    @operation_monitoring(operation_name="pipeline_initialization")
    async def initialize(self) -> bool:
        """Initialize the enhanced market analysis pipeline."""
        start_time = time.time()
        self.logger.info("🚀 Initializing Enhanced Market Analysis Pipeline...")
        
        try:
            # Initialize data quality framework
            await self.data_quality.initialize()
            
            # Initialize security framework
            await self.security.initialize()
            
            # Initialize all pipeline steps
            for step_name, step in self.steps.items():
                self.logger.info(f"🔧 Initializing {step_name}...")
                step_success = await step.initialize()
                if not step_success:
                    self.logger.error(f"❌ Failed to initialize {step_name}")
                    return False
            
            # Log initialization success
            duration = time.time() - start_time
            self.pipeline_timings['initialization'] = duration
            self.logger.info(f"✅ Enhanced Market Analysis Pipeline initialized successfully in {duration:.3f}s")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize pipeline: {e}")
            return False
    
    @handles_errors(Exception, fallback=False, context="pipeline_execution")
    @data_protection(operation_type="pipeline_execution")
    @operation_monitoring(operation_name="pipeline_execution")
    async def execute(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the enhanced market analysis pipeline with comprehensive validation."""
        start_time = time.time()
        self.logger.info("🎯 Executing Enhanced Market Analysis Pipeline...")
        
        try:
            # Validate training input
            input_validation = await self._validate_training_input(training_input)
            if not input_validation.get('valid', False):
                return {
                    'success': False,
                    'error': f"Training input validation failed: {input_validation.get('error')}",
                    'timestamp': get_current_datetime().isoformat()
                }
            
            # Execute pipeline steps in sequence
            step_order = ['data_collection', 'hmm_clustering', 'feature_engineering']
            
            for step_name in step_order:
                if step_name not in self.steps:
                    self.logger.warning(f"⚠️ Step {step_name} not found, skipping...")
                    continue
                
                self.logger.info(f"🔄 Executing {step_name}...")
                step = self.steps[step_name]
                
                # Execute step
                step_result = await step.execute(training_input, self.pipeline_state)
                self.execution_results[step_name] = step_result
                
                # Update pipeline state
                self.pipeline_state[step_name] = step_result
                
                # Check if step failed
                if not step_result.get('success', False):
                    self.logger.error(f"❌ Step {step_name} failed: {step_result.get('error')}")
                    return {
                        'success': False,
                        'error': f"Pipeline failed at step {step_name}: {step_result.get('error')}",
                        'failed_step': step_name,
                        'execution_results': self.execution_results,
                        'timestamp': get_current_datetime().isoformat()
                    }
                
                self.logger.info(f"✅ Step {step_name} completed successfully")
            
            # Log execution success
            duration = time.time() - start_time
            self.pipeline_timings['execution'] = duration
            
            self.logger.info("🎉 Enhanced Market Analysis Pipeline completed successfully!")
            return {
                'success': True,
                'execution_results': self.execution_results,
                'pipeline_state': self.pipeline_state,
                'execution_time': duration,
                'timestamp': get_current_datetime().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
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
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline execution summary."""
        return {
            'pipeline_name': 'Enhanced Market Analysis Pipeline',
            'execution_results': self.execution_results,
            'pipeline_state': self.pipeline_state,
            'timings': self.pipeline_timings,
            'timestamp': get_current_datetime().isoformat()
        }


# Main pipeline execution function
async def run_enhanced_market_analysis_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str = '1m',
    data_dir: str = 'data_cache',
    **config
) -> Dict[str, Any]:
    """Run the enhanced market analysis pipeline with comprehensive validation and protection."""
    
    # Prepare training input
    training_input = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'timestamp': get_current_datetime().isoformat()
    }
    
    # Initialize pipeline
    pipeline = EnhancedMarketAnalysisPipeline(config)
    
    # Initialize pipeline
    init_success = await pipeline.initialize()
    if not init_success:
        return {
            'success': False,
            'error': 'Failed to initialize pipeline',
            'timestamp': get_current_datetime().isoformat()
        }
    
    # Execute pipeline
    result = await pipeline.execute(training_input)
    
    # Add pipeline summary
    if result.get('success', False):
        result['pipeline_summary'] = pipeline.get_pipeline_summary()
    
    return result


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'force_rerun': True,
            'hmm_clustering': True,
            'feature_engineering': True,
            'random_state': 42,
        }
        
        result = await run_enhanced_market_analysis_pipeline(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache',
            **config
        )
        
        print(f"Pipeline result: {result}")
    
    asyncio.run(main())