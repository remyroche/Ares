#!/usr/bin/env python3
"""Enhanced Main Orchestrator for All Training Pipelines.

This module provides a comprehensive interface to run all training pipelines with:
- Comprehensive error handling and validation
- Step-by-step data validation and quality checks
- Decorators for data formatting, analysis, and access protection
- Rollback and recovery mechanisms
- Enhanced logging and monitoring

Pipelines:
1. Data Collection Pipeline
2. Market Analysis Pipeline
3. Model Training Pipeline
4. Optimization Pipeline
5. Backtesting Pipeline
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced utilities and decorators
from src.core.domain import handle_errors, memory_efficient, validate_data_quality, monitor_pipeline_step
from src.utils.common_operations import (
    format_datetime, get_current_datetime, ensure_directory, 
    safe_json_dump, safe_json_load, safe_file_exists
)
from src.utils.validator_orchestrator import ValidatorOrchestrator
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.pipeline_standards import pipeline_standards, PipelineStandards
from src.utils.logger import system_logger
from src.utils.prometheus_metrics import metrics
from src.utils.enhanced_memory_management import memory_efficient as memory_efficient_util
from src.utils.data_formatting_framework import DataFormattingFramework

# Import all pipeline modules
from src.training.steps.data_collection import run_data_collection_pipeline
from src.training.steps.market_analysis import run_market_analysis_pipeline
from src.training.steps.model_training import run_model_training_pipeline
from src.training.steps.optimisation import run_optimisation_pipeline
from src.training.steps.backtesting import run_backtesting_pipeline


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    name: str
    status: PipelineStatus
    execution_time: float
    error: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[float] = None
    rollback_required: bool = False


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    force_rerun: bool = True
    enable_validation: bool = True
    enable_rollback: bool = True
    enable_monitoring: bool = True
    validation_level: str = "CRITICAL"
    max_retries: int = 3
    timeout_seconds: int = 3600


class EnhancedPipelineOrchestrator:
    """Enhanced orchestrator for all training pipelines with comprehensive validation and error handling."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = system_logger.getChild("EnhancedPipelineOrchestrator")
        self.validator_orchestrator = ValidatorOrchestrator()
        self.data_quality_framework = DataQualityFramework()
        self.data_formatting_framework = DataFormattingFramework()
        self.pipeline_results: List[PipelineResult] = []
        self.pipeline_state: Dict[str, Any] = {}
        self.checkpoint_file = Path(config.data_dir) / f"pipeline_checkpoint_{config.symbol}_{config.timeframe}.json"
        
        # Ensure data directory exists
        ensure_directory(config.data_dir)
        
        # Initialize monitoring
        if config.enable_monitoring:
            self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize monitoring and metrics collection."""
        try:
            metrics.initialize_pipeline_monitoring(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe
            )
            self.logger.info("✅ Pipeline monitoring initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize monitoring: {e}")
    
    @handle_errors(fallback=False)
    @monitor_pipeline_step("prerequisites_validation")
    async def validate_pipeline_prerequisites(self) -> bool:
        """Validate prerequisites before starting pipeline execution."""
        self.logger.info("🔍 Validating pipeline prerequisites...")
        
        try:
            # Validate configuration using pipeline standards
            if not self.config.symbol or not self.config.exchange:
                self.logger.error("❌ Invalid symbol or exchange configuration")
                return False
            
            # Validate data directory using pipeline standards
            data_dir_path = PipelineStandards.build_path(
                "raw_data", 
                self.config.exchange, 
                self.config.symbol
            )
            if not safe_file_exists(data_dir_path) and not safe_file_exists(self.config.data_dir):
                self.logger.error(f"❌ Data directory does not exist: {self.config.data_dir}")
                return False
            
            # Validate environment dependencies using pipeline standards
            required_modules = ['pandas', 'numpy', 'asyncio', 'pathlib']
            dependency_check = PipelineStandards.validate_environment_dependencies(
                required_modules, self.logger
            )
            
            missing_deps = [mod for mod, available in dependency_check.items() if not available]
            if missing_deps:
                self.logger.error(f"❌ Missing required dependencies: {missing_deps}")
                return False
            
            # Validate pipeline standards compliance
            standards_check = await pipeline_standards.validate_environment()
            if not standards_check.get("passed", False):
                self.logger.error(f"❌ Pipeline standards validation failed: {standards_check.get('error')}")
                return False
            
            self.logger.info("✅ Pipeline prerequisites validated successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            return False
    
    @handle_errors(fallback=None)
    @monitor_pipeline_step("save_checkpoint")
    async def save_checkpoint(self, pipeline_name: str, result: PipelineResult):
        """Save pipeline execution checkpoint."""
        try:
            checkpoint_data = {
                "timestamp": format_datetime(get_current_datetime()),
                "pipeline_name": pipeline_name,
                "result": {
                    "name": result.name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error": result.error,
                    "data_quality_score": result.data_quality_score
                },
                "pipeline_state": self.pipeline_state,
                "config": {
                    "symbol": self.config.symbol,
                    "exchange": self.config.exchange,
                    "timeframe": self.config.timeframe,
                    "data_dir": self.config.data_dir
                }
            }
            
            safe_json_dump(checkpoint_data, self.checkpoint_file)
            self.logger.debug(f"💾 Checkpoint saved for {pipeline_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save checkpoint: {e}")
    
    @handle_errors(fallback=None)
    @monitor_pipeline_step("load_checkpoint")
    async def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load pipeline execution checkpoint."""
        try:
            if safe_file_exists(self.checkpoint_file):
                checkpoint_data = safe_json_load(self.checkpoint_file)
                self.logger.info(f"📂 Checkpoint loaded from {self.checkpoint_file}")
                return checkpoint_data
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    @handle_errors(fallback=False)
    @validate_data_quality()
    @monitor_pipeline_step("validate_pipeline_data")
    async def validate_pipeline_data(self, pipeline_name: str, data_paths: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Validate data quality and structure for a pipeline."""
        self.logger.info(f"🔍 Validating data for {pipeline_name}...")
        
        try:
            validation_results = {}
            all_passed = True
            
            for data_path in data_paths:
                if not safe_file_exists(data_path):
                    self.logger.error(f"❌ Data file not found: {data_path}")
                    validation_results[data_path] = {"exists": False, "error": "File not found"}
                    all_passed = False
                    continue
                
                # Use pipeline standards for file validation
                file_type = self._determine_file_type(data_path)
                schema_validation = PipelineStandards.validate_file_schema(
                    file_path=data_path,
                    expected_schema=file_type,
                    logger=self.logger
                )
                
                # Validate data quality using pipeline standards
                quality_result = await self.data_quality_framework.validate_data_file(
                    file_path=data_path,
                    validation_level=self.config.validation_level
                )
                
                # Combine validation results
                combined_result = {
                    "schema_validation": schema_validation,
                    "quality_validation": quality_result,
                    "overall_passed": schema_validation.get("passed", False) and quality_result.get("passed", False)
                }
                
                validation_results[data_path] = combined_result
                
                if not combined_result["overall_passed"]:
                    all_passed = False
                    self.logger.warning(f"⚠️ Data validation issues in {data_path}")
                    if not schema_validation.get("passed", False):
                        self.logger.warning(f"   Schema issues: {schema_validation.get('error')}")
                    if not quality_result.get("passed", False):
                        self.logger.warning(f"   Quality issues: {quality_result.get('error')}")
            
            if all_passed:
                self.logger.info(f"✅ Data validation passed for {pipeline_name}")
            else:
                self.logger.warning(f"⚠️ Data validation issues found for {pipeline_name}")
            
            return all_passed, validation_results
            
        except Exception as e:
            self.logger.exception(f"❌ Data validation failed for {pipeline_name}: {e}")
            return False, {"error": str(e)}
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type based on path for schema validation."""
        if "aggtrades" in file_path:
            return "aggtrades"
        elif "klines" in file_path:
            return "klines"
        elif "futures" in file_path:
            return "futures"
        elif "unified" in file_path:
            return "unified"
        else:
            return "general"
    
    @handle_errors(fallback=False)
    @monitor_pipeline_step("rollback_pipeline")
    async def rollback_pipeline(self, pipeline_name: str) -> bool:
        """Rollback pipeline execution if needed."""
        if not self.config.enable_rollback:
            self.logger.info(f"⏭️ Rollback disabled for {pipeline_name}")
            return True
        
        self.logger.info(f"🔄 Rolling back {pipeline_name}...")
        
        try:
            # Load checkpoint to get previous state
            checkpoint = await self.load_checkpoint()
            if checkpoint:
                # Restore previous pipeline state
                self.pipeline_state = checkpoint.get("pipeline_state", {})
                self.logger.info(f"✅ Rollback completed for {pipeline_name}")
                return True
            else:
                self.logger.warning(f"⚠️ No checkpoint found for rollback of {pipeline_name}")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Rollback failed for {pipeline_name}: {e}")
            return False
    
    @memory_efficient
    @handle_errors(fallback=False)
    @monitor_pipeline_step("execute_pipeline")
    async def execute_pipeline(self, pipeline_name: str, pipeline_func, pipeline_config: Dict[str, Any]) -> PipelineResult:
        """Execute a single pipeline with comprehensive error handling and validation."""
        self.logger.info(f"🚀 Executing {pipeline_name} pipeline...")
        
        start_time = time.time()
        result = PipelineResult(
            name=pipeline_name,
            status=PipelineStatus.RUNNING,
            execution_time=0.0
        )
        
        try:
            # Validate prerequisites
            if not await self.validate_pipeline_prerequisites():
                result.status = PipelineStatus.FAILED
                result.error = "Prerequisites validation failed"
                return result
            
            # Execute pipeline with timeout
            pipeline_task = asyncio.create_task(
                pipeline_func(
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe,
                    data_dir=self.config.data_dir,
                    **pipeline_config
                )
            )
            
            success = await asyncio.wait_for(pipeline_task, timeout=self.config.timeout_seconds)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            if success:
                result.status = PipelineStatus.COMPLETED
                self.logger.info(f"✅ {pipeline_name} completed successfully in {execution_time:.2f}s")
                
                # Record metrics
                if self.config.enable_monitoring:
                    metrics.record_pipeline_execution(
                        pipeline_name=pipeline_name,
                        duration=execution_time,
                        status="SUCCESS"
                    )
            else:
                result.status = PipelineStatus.FAILED
                result.error = "Pipeline execution returned False"
                result.rollback_required = True
                self.logger.error(f"❌ {pipeline_name} failed after {execution_time:.2f}s")
                
                # Record metrics
                if self.config.enable_monitoring:
                    metrics.record_pipeline_execution(
                        pipeline_name=pipeline_name,
                        duration=execution_time,
                        status="FAILED"
                    )
            
            # Save checkpoint
            await self.save_checkpoint(pipeline_name, result)
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.status = PipelineStatus.FAILED
            result.error = f"Pipeline timeout after {self.config.timeout_seconds}s"
            result.rollback_required = True
            self.logger.error(f"⏰ {pipeline_name} timed out after {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            result.rollback_required = True
            self.logger.exception(f"💥 {pipeline_name} failed with exception: {e}")
        
        return result

    @handle_errors(fallback=False)
    @monitor_pipeline_step("run_all_pipelines")
    async def run_all_pipelines(self) -> bool:
        """Run all training pipelines in sequence with enhanced validation and error handling."""
        
        self.logger.info("🚀 ENHANCED COMPLETE TRADING PIPELINE EXECUTION")
        self.logger.info("=" * 100)
        self.logger.info(f"📊 Configuration:")
        self.logger.info(f"   Symbol: {self.config.symbol}")
        self.logger.info(f"   Exchange: {self.config.exchange}")
        self.logger.info(f"   Timeframe: {self.config.timeframe}")
        self.logger.info(f"   Data directory: {self.config.data_dir}")
        self.logger.info(f"   Validation level: {self.config.validation_level}")
        self.logger.info(f"   Enable rollback: {self.config.enable_rollback}")
        self.logger.info(f"   Enable monitoring: {self.config.enable_monitoring}")
        self.logger.info("=" * 100)
        
        total_start_time = time.time()
        
        # Pipeline configurations with enhanced validation
        pipeline_configs = {
            'data_collection': {
                'force_rerun': self.config.force_rerun,
                'quality_checks': True,
                'validate_data': True,
                'convert_format': True,
                'validation_level': self.config.validation_level,
            },
            'market_analysis': {
                'force_rerun': self.config.force_rerun,
                'hmm_clustering': True,
                'regime_splitting': True,
                'feature_engineering': True,
                'matrix_operations': True,
                'feature_selection': True,
                'validation_level': self.config.validation_level,
            },
            'model_training': {
                'force_rerun': self.config.force_rerun,
                'hmm_training': True,
                'regime_intelligence': True,
                'analyst_creation': True,
                'analyst_enhancement': True,
                'ensemble_creation': True,
                'tactician_training': True,
                'validation_level': self.config.validation_level,
            },
            'optimisation': {
                'force_rerun': self.config.force_rerun,
                'confidence_calibration': True,
                'parameter_optimization': True,
                'validation_level': self.config.validation_level,
            },
            'backtesting': {
                'force_rerun': self.config.force_rerun,
                'walk_forward_validation': True,
                'monte_carlo_validation': True,
                'ab_testing': True,
                'model_saving': True,
                'validation_level': self.config.validation_level,
            }
        }
        
        # Pipeline execution order with data dependencies using pipeline standards
        pipelines = [
            ('Data Collection', run_data_collection_pipeline, pipeline_configs['data_collection'], [
                PipelineStandards.build_path("raw_data", self.config.exchange, self.config.symbol) + 
                f"/{PipelineStandards.FILE_NAMING['aggtrades'].format(exchange=self.config.exchange, asset=self.config.symbol)}"
            ]),
            ('Market Analysis', run_market_analysis_pipeline, pipeline_configs['market_analysis'], [
                PipelineStandards.build_path("raw_data", self.config.exchange, self.config.symbol) + 
                f"/{PipelineStandards.FILE_NAMING['aggtrades'].format(exchange=self.config.exchange, asset=self.config.symbol)}",
                PipelineStandards.build_path("processed_data", self.config.exchange, self.config.symbol) + 
                f"/volume_{self.config.exchange}_{self.config.symbol}_consolidated.parquet"
            ]),
            ('Model Training', run_model_training_pipeline, pipeline_configs['model_training'], [
                PipelineStandards.build_path("raw_data", self.config.exchange, self.config.symbol) + 
                f"/{PipelineStandards.FILE_NAMING['aggtrades'].format(exchange=self.config.exchange, asset=self.config.symbol)}",
                f"models/{self.config.symbol}_{self.config.exchange}_hmm_model.pkl"
            ]),
            ('Optimization', run_optimisation_pipeline, pipeline_configs['optimisation'], [
                f"models/{self.config.symbol}_{self.config.exchange}_analyst_ensemble.pkl",
                f"models/{self.config.symbol}_{self.config.exchange}_tactician_model.pkl"
            ]),
            ('Backtesting', run_backtesting_pipeline, pipeline_configs['backtesting'], [
                PipelineStandards.build_path("raw_data", self.config.exchange, self.config.symbol) + 
                f"/{PipelineStandards.FILE_NAMING['aggtrades'].format(exchange=self.config.exchange, asset=self.config.symbol)}",
                f"models/{self.config.symbol}_{self.config.exchange}_final_models.pkl"
            ]),
        ]
        
        # Execute pipelines in sequence with validation
        for pipeline_name, pipeline_func, pipeline_config, data_dependencies in pipelines:
            self.logger.info(f"\n🔄 Starting {pipeline_name} Pipeline...")
            self.logger.info("-" * 80)
            
            # Validate data dependencies
            if self.config.enable_validation:
                data_valid, validation_results = await self.validate_pipeline_data(
                    pipeline_name, data_dependencies
                )
                if not data_valid:
                    self.logger.error(f"❌ Data validation failed for {pipeline_name}")
                    # Continue with warning if not critical
                    if self.config.validation_level == "CRITICAL":
                        return False
            
            # Execute pipeline
            result = await self.execute_pipeline(pipeline_name, pipeline_func, pipeline_config)
            self.pipeline_results.append(result)
            
            # Handle rollback if needed
            if result.rollback_required and self.config.enable_rollback:
                rollback_success = await self.rollback_pipeline(pipeline_name)
                if not rollback_success:
                    self.logger.error(f"❌ Rollback failed for {pipeline_name}")
                    return False
                result.status = PipelineStatus.ROLLED_BACK
        
        # Final results
        total_time = time.time() - total_start_time
        
        self.logger.info("\n" + "=" * 100)
        self.logger.info("📊 ENHANCED FINAL RESULTS SUMMARY")
        self.logger.info("=" * 100)
        
        successful_pipelines = 0
        failed_pipelines = 0
        rolled_back_pipelines = 0
        
        for result in self.pipeline_results:
            status_emoji = {
                PipelineStatus.COMPLETED: "✅ SUCCESS",
                PipelineStatus.FAILED: "❌ FAILED",
                PipelineStatus.ROLLED_BACK: "🔄 ROLLED_BACK"
            }.get(result.status, "❓ UNKNOWN")
            
            self.logger.info(f"{result.name:20} | {status_emoji:15} | {result.execution_time:8.2f}s")
            if result.error:
                self.logger.info(f"{'':20} | Error: {result.error}")
            if result.data_quality_score is not None:
                self.logger.info(f"{'':20} | Quality Score: {result.data_quality_score:.3f}")
            
            if result.status == PipelineStatus.COMPLETED:
                successful_pipelines += 1
            elif result.status == PipelineStatus.FAILED:
                failed_pipelines += 1
            elif result.status == PipelineStatus.ROLLED_BACK:
                rolled_back_pipelines += 1
        
        self.logger.info("-" * 100)
        self.logger.info(f"Total Execution Time: {total_time:.2f} seconds")
        self.logger.info(f"Successful Pipelines: {successful_pipelines}/{len(pipelines)}")
        self.logger.info(f"Failed Pipelines: {failed_pipelines}/{len(pipelines)}")
        self.logger.info(f"Rolled Back Pipelines: {rolled_back_pipelines}/{len(pipelines)}")
        
        if failed_pipelines == 0 and rolled_back_pipelines == 0:
            self.logger.info("🎉 ALL PIPELINES COMPLETED SUCCESSFULLY!")
        else:
            self.logger.info(f"⚠️  {failed_pipelines} PIPELINE(S) FAILED, {rolled_back_pipelines} ROLLED BACK")
        
        self.logger.info("=" * 100)
        
        # Save enhanced results using pipeline standards
        results_dir = PipelineStandards.build_path("reports", self.config.exchange, self.config.symbol)
        ensure_directory(results_dir)
        results_file = Path(results_dir) / f"enhanced_pipeline_results_{self.config.symbol}_{self.config.timeframe}_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.json"
        enhanced_results = {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe,
            'total_execution_time': total_time,
            'successful_pipelines': successful_pipelines,
            'failed_pipelines': failed_pipelines,
            'rolled_back_pipelines': rolled_back_pipelines,
            'pipeline_results': [
                {
                    'name': result.name,
                    'status': result.status.value,
                    'execution_time': result.execution_time,
                    'error': result.error,
                    'data_quality_score': result.data_quality_score,
                    'rollback_required': result.rollback_required
                }
                for result in self.pipeline_results
            ],
            'config': {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'data_dir': self.config.data_dir,
                'validation_level': self.config.validation_level,
                'enable_rollback': self.config.enable_rollback,
                'enable_monitoring': self.config.enable_monitoring
            },
            'timestamp': format_datetime(get_current_datetime())
        }
        
        safe_json_dump(enhanced_results, results_file)
        self.logger.info(f"💾 Enhanced results saved to: {results_file}")
        
        return failed_pipelines == 0


# Legacy function for backward compatibility
async def run_all_pipelines(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **config: Dict[str, Any]
) -> bool:
    """Legacy function for backward compatibility. Use EnhancedPipelineOrchestrator for new implementations."""
    
    # Create enhanced configuration
    pipeline_config = PipelineConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=config.get('force_rerun', True),
        enable_validation=config.get('enable_validation', True),
        enable_rollback=config.get('enable_rollback', True),
        enable_monitoring=config.get('enable_monitoring', True),
        validation_level=config.get('validation_level', 'CRITICAL'),
        max_retries=config.get('max_retries', 3),
        timeout_seconds=config.get('timeout_seconds', 3600)
    )
    
    # Create and run enhanced orchestrator
    orchestrator = EnhancedPipelineOrchestrator(pipeline_config)
    return await orchestrator.run_all_pipelines()

async def main():
    """Enhanced main function to run all pipelines with comprehensive validation and error handling."""
    
    # Enhanced configuration with validation and monitoring
    pipeline_config = PipelineConfig(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_dir="data_cache",
        force_rerun=True,
        enable_validation=True,
        enable_rollback=True,
        enable_monitoring=True,
        validation_level="CRITICAL",
        max_retries=3,
        timeout_seconds=3600
    )
    
    # Create enhanced orchestrator
    orchestrator = EnhancedPipelineOrchestrator(pipeline_config)
    
    try:
        success = await orchestrator.run_all_pipelines()
        
        if success:
            print("\n🎉 ENHANCED COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
            print("✅ All pipelines completed with comprehensive validation and error handling")
            sys.exit(0)
        else:
            print("\n❌ ENHANCED PIPELINE EXECUTION FAILED!")
            print("💥 Some pipelines failed - check logs for details")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR in enhanced pipeline execution: {e}")
        print("🔍 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    # Run all pipelines
    asyncio.run(main())