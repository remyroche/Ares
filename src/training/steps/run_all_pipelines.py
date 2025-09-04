#!/usr/bin/env python3
"""Enhanced Main Orchestrator for All Training Pipelines.

This module provides the main interface to run all training pipelines with comprehensive
validation, error handling, and monitoring:
1. Data Collection Pipeline
2. Market Analysis Pipeline
3. Model Training Pipeline
4. Optimization Pipeline
5. Backtesting Pipeline

Enhanced with:
- Comprehensive validation at each step
- Data integrity checks
- Performance monitoring
- Error recovery mechanisms
- State management and checkpointing
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced pipeline utilities
from src.utils.pipeline_validator_framework import (
    validator_orchestrator,
    ValidationLevel,
    ValidationResult
)
from src.utils.pipeline_decorators import (
    pipeline_step,
    step_dependency_check,
    performance_monitor
)
from src.utils.pipeline_utilities import (
    pipeline_utilities,
    DataFormat,
    DataMetadata
)
from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.logging import logs_execution
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory
)

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
    SKIPPED = "skipped"


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    name: str
    status: PipelineStatus
    execution_time: float
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    data_quality_metrics: Dict[str, Any] = field(default_factory=dict)


class EnhancedPipelineOrchestrator:
    """Enhanced orchestrator for all training pipelines with comprehensive validation."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str = "1m", data_dir: str = "data_cache"):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.data_dir = data_dir
        self.logger = logging.getLogger("enhanced_pipeline_orchestrator")
        self.results: List[PipelineResult] = []
        self.checkpoint_file = Path(data_dir) / f"pipeline_checkpoint_{symbol}_{timeframe}.json"
        
        # Setup logging
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    @handles_errors(Exception, fallback=False)
    @logs_execution("enhanced_pipeline_orchestrator")
    async def run_all_pipelines(self, **config: Dict[str, Any]) -> bool:
        """Run all training pipelines with enhanced validation and monitoring."""
        
        self.logger.info("🚀 ENHANCED COMPLETE TRADING PIPELINE EXECUTION")
        self.logger.info("=" * 100)
        self.logger.info(f"📊 Configuration:")
        self.logger.info(f"   Symbol: {self.symbol}")
        self.logger.info(f"   Exchange: {self.exchange}")
        self.logger.info(f"   Timeframe: {self.timeframe}")
        self.logger.info(f"   Data directory: {self.data_dir}")
        self.logger.info("=" * 100)
        
        total_start_time = time.time()
        
        # Load checkpoint if exists
        await self._load_checkpoint()
        
        # Pipeline configurations with enhanced validation
        pipeline_configs = {
            'data_collection': {
                'force_rerun': config.get('force_rerun', True),
                'quality_checks': config.get('quality_checks', True),
                'validate_data': config.get('validate_data', True),
                'convert_format': config.get('convert_format', True),
                'validation_level': ValidationLevel.CRITICAL,
                'enable_monitoring': True,
            },
            'market_analysis': {
                'force_rerun': config.get('force_rerun', True),
                'hmm_clustering': config.get('hmm_clustering', True),
                'regime_splitting': config.get('regime_splitting', True),
                'feature_engineering': config.get('feature_engineering', True),
                'matrix_operations': config.get('matrix_operations', True),
                'feature_selection': config.get('feature_selection', True),
                'validation_level': ValidationLevel.CRITICAL,
                'enable_monitoring': True,
            },
            'model_training': {
                'force_rerun': config.get('force_rerun', True),
                'hmm_training': config.get('hmm_training', True),
                'regime_intelligence': config.get('regime_intelligence', True),
                'analyst_creation': config.get('analyst_creation', True),
                'analyst_enhancement': config.get('analyst_enhancement', True),
                'ensemble_creation': config.get('ensemble_creation', True),
                'tactician_training': config.get('tactician_training', True),
                'validation_level': ValidationLevel.CRITICAL,
                'enable_monitoring': True,
            },
            'optimisation': {
                'force_rerun': config.get('force_rerun', True),
                'confidence_calibration': config.get('confidence_calibration', True),
                'parameter_optimization': config.get('parameter_optimization', True),
                'validation_level': ValidationLevel.CRITICAL,
                'enable_monitoring': True,
            },
            'backtesting': {
                'force_rerun': config.get('force_rerun', True),
                'walk_forward_validation': config.get('walk_forward_validation', True),
                'monte_carlo_validation': config.get('monte_carlo_validation', True),
                'ab_testing': config.get('ab_testing', True),
                'model_saving': config.get('model_saving', True),
                'validation_level': ValidationLevel.CRITICAL,
                'enable_monitoring': True,
            }
        }
        
        # Pipeline execution order with enhanced validation
        pipelines = [
            ('Data Collection', run_data_collection_pipeline, pipeline_configs['data_collection']),
            ('Market Analysis', run_market_analysis_pipeline, pipeline_configs['market_analysis']),
            ('Model Training', run_model_training_pipeline, pipeline_configs['model_training']),
            ('Optimization', run_optimisation_pipeline, pipeline_configs['optimisation']),
            ('Backtesting', run_backtesting_pipeline, pipeline_configs['backtesting']),
        ]
        
        # Execute pipelines with enhanced monitoring
        for pipeline_name, pipeline_func, pipeline_config in pipelines:
            await self._execute_pipeline_with_validation(
                pipeline_name, pipeline_func, pipeline_config
            )
            
            # Save checkpoint after each pipeline
            await self._save_checkpoint()
        
        # Final results and reporting
        total_time = time.time() - total_start_time
        await self._generate_final_report(total_time, config)
        
        # Clean up checkpoint file on success
        if self._all_pipelines_successful():
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.logger.info("Checkpoint file cleaned up after successful completion")
        
        return self._all_pipelines_successful()
    
    @handles_errors(Exception, fallback=False)
    async def _execute_pipeline_with_validation(
        self,
        pipeline_name: str,
        pipeline_func: Callable,
        pipeline_config: Dict[str, Any]
    ) -> None:
        """Execute a single pipeline with comprehensive validation."""
        
        self.logger.info(f"\n🔄 Starting {pipeline_name} Pipeline...")
        self.logger.info("-" * 80)
        
        pipeline_start_time = time.time()
        start_timestamp = format_datetime(get_current_datetime())
        
        # Create pipeline result
        result = PipelineResult(
            name=pipeline_name,
            status=PipelineStatus.RUNNING,
            execution_time=0.0,
            start_time=start_timestamp
        )
        
        try:
            # Pre-execution validation
            await self._validate_pipeline_prerequisites(pipeline_name, pipeline_config)
            
            # Execute pipeline with monitoring
            with pipeline_utilities.safe_data_operation(
                f"pipeline_{pipeline_name.lower().replace(' ', '_')}",
                self._get_pipeline_output_file(pipeline_name)
            ):
                success = await pipeline_func(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    data_dir=self.data_dir,
                    **pipeline_config
                )
            
            # Post-execution validation
            validation_results = await self._validate_pipeline_output(pipeline_name)
            performance_metrics = await self._collect_performance_metrics(pipeline_name)
            data_quality_metrics = await self._collect_data_quality_metrics(pipeline_name)
            
            # Update result
            pipeline_time = time.time() - pipeline_start_time
            result.status = PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
            result.execution_time = pipeline_time
            result.end_time = format_datetime(get_current_datetime())
            result.validation_results = validation_results
            result.performance_metrics = performance_metrics
            result.data_quality_metrics = data_quality_metrics
            
            if not success:
                result.error = f"Pipeline {pipeline_name} returned False"
            
            self.logger.info(f"✅ {pipeline_name} Pipeline completed successfully in {pipeline_time:.2f} seconds")
            
        except Exception as e:
            pipeline_time = time.time() - pipeline_start_time
            result.status = PipelineStatus.FAILED
            result.execution_time = pipeline_time
            result.end_time = format_datetime(get_current_datetime())
            result.error = str(e)
            
            self.logger.error(f"💥 {pipeline_name} Pipeline failed with exception: {e}")
            self.logger.error(f"⏱️ Execution time: {pipeline_time:.2f} seconds")
        
        # Add result to list
        self.results.append(result)
    
    @handles_errors(Exception, fallback={})
    async def _validate_pipeline_prerequisites(
        self,
        pipeline_name: str,
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate prerequisites before pipeline execution."""
        
        validation_results = await validator_orchestrator.validate_pipeline_step(
            step_name=f"{pipeline_name}_prerequisites",
            data=None,
            context={
                "pipeline_name": pipeline_name,
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": self.timeframe,
                "data_dir": self.data_dir,
                "config": pipeline_config
            },
            validators_to_run=["step_dependency"]
        )
        
        # Check if validation failed
        for validator_name, report in validation_results.items():
            if report.result == ValidationResult.FAILED:
                raise ValueError(f"Prerequisites validation failed for {pipeline_name}: {report.message}")
            elif report.result == ValidationResult.WARNING:
                self.logger.warning(f"Prerequisites validation warning for {pipeline_name}: {report.message}")
        
        return validation_results
    
    @handles_errors(Exception, fallback={})
    async def _validate_pipeline_output(self, pipeline_name: str) -> Dict[str, Any]:
        """Validate pipeline output after execution."""
        
        output_file = self._get_pipeline_output_file(pipeline_name)
        if not safe_file_exists(output_file):
            return {"error": f"Output file not found: {output_file}"}
        
        try:
            # Load output data for validation
            output_data = pipeline_utilities.format_manager.read_data(output_file)
            
            validation_results = await validator_orchestrator.validate_pipeline_step(
                step_name=f"{pipeline_name}_output",
                data=output_data,
                context={
                    "pipeline_name": pipeline_name,
                    "output_file": output_file,
                    "validation_level": ValidationLevel.CRITICAL
                },
                validators_to_run=["data_format", "data_quality"]
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"Could not validate output for {pipeline_name}: {e}")
            return {"error": str(e)}
    
    @handles_errors(Exception, fallback={})
    async def _collect_performance_metrics(self, pipeline_name: str) -> Dict[str, Any]:
        """Collect performance metrics for the pipeline."""
        
        # Get validation summary
        validation_summary = validator_orchestrator.get_validation_summary()
        
        return {
            "validation_summary": validation_summary,
            "pipeline_utilities_status": pipeline_utilities.get_pipeline_status(),
            "timestamp": format_datetime(get_current_datetime())
        }
    
    @handles_errors(Exception, fallback={})
    async def _collect_data_quality_metrics(self, pipeline_name: str) -> Dict[str, Any]:
        """Collect data quality metrics for the pipeline."""
        
        output_file = self._get_pipeline_output_file(pipeline_name)
        if not safe_file_exists(output_file):
            return {"error": f"Output file not found: {output_file}"}
        
        try:
            # Get data metadata
            metadata = pipeline_utilities.format_manager.get_data_metadata(output_file)
            
            # Analyze data quality
            output_data = pipeline_utilities.format_manager.read_data(output_file)
            quality_analysis = pipeline_utilities.analysis_manager.analyze_data_quality(output_data)
            
            return {
                "metadata": metadata.to_dict(),
                "quality_analysis": quality_analysis
            }
            
        except Exception as e:
            self.logger.warning(f"Could not collect data quality metrics for {pipeline_name}: {e}")
            return {"error": str(e)}
    
    def _get_pipeline_output_file(self, pipeline_name: str) -> str:
        """Get the expected output file for a pipeline."""
        
        pipeline_outputs = {
            "Data Collection": f"{self.data_dir}/aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet",
            "Market Analysis": f"{self.data_dir}/market_analysis_{self.symbol}_{self.timeframe}.parquet",
            "Model Training": f"{self.data_dir}/model_training_{self.symbol}_{self.timeframe}.parquet",
            "Optimization": f"{self.data_dir}/optimization_{self.symbol}_{self.timeframe}.parquet",
            "Backtesting": f"{self.data_dir}/backtesting_{self.symbol}_{self.timeframe}.parquet",
        }
        
        return pipeline_outputs.get(pipeline_name, f"{self.data_dir}/{pipeline_name.lower().replace(' ', '_')}.parquet")
    
    @handles_errors(Exception, fallback=False)
    async def _load_checkpoint(self) -> None:
        """Load pipeline checkpoint if it exists."""
        
        if self.checkpoint_file.exists():
            try:
                checkpoint_data = safe_json_load(self.checkpoint_file)
                self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                
                # Restore results from checkpoint
                for result_data in checkpoint_data.get("results", []):
                    result = PipelineResult(
                        name=result_data["name"],
                        status=PipelineStatus(result_data["status"]),
                        execution_time=result_data["execution_time"],
                        start_time=result_data["start_time"],
                        end_time=result_data.get("end_time"),
                        error=result_data.get("error"),
                        validation_results=result_data.get("validation_results", {}),
                        performance_metrics=result_data.get("performance_metrics", {}),
                        data_quality_metrics=result_data.get("data_quality_metrics", {})
                    )
                    self.results.append(result)
                
                self.logger.info(f"Restored {len(self.results)} pipeline results from checkpoint")
                
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    @handles_errors(Exception, fallback=None)
    async def _save_checkpoint(self) -> None:
        """Save pipeline checkpoint."""
        
        try:
            checkpoint_data = {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": self.timeframe,
                "data_dir": self.data_dir,
                "timestamp": format_datetime(get_current_datetime()),
                "results": [
                    {
                        "name": result.name,
                        "status": result.status.value,
                        "execution_time": result.execution_time,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "error": result.error,
                        "validation_results": result.validation_results,
                        "performance_metrics": result.performance_metrics,
                        "data_quality_metrics": result.data_quality_metrics
                    }
                    for result in self.results
                ]
            }
            
            ensure_directory(Path(self.checkpoint_file).parent)
            safe_json_dump(checkpoint_data, self.checkpoint_file, indent=2)
            self.logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint: {e}")
    
    @handles_errors(Exception, fallback=None)
    async def _generate_final_report(self, total_time: float, config: Dict[str, Any]) -> None:
        """Generate comprehensive final report."""
        
        self.logger.info("\n" + "=" * 100)
        self.logger.info("📊 ENHANCED FINAL RESULTS SUMMARY")
        self.logger.info("=" * 100)
        
        successful_pipelines = 0
        failed_pipelines = 0
        
        for result in self.results:
            status = "✅ SUCCESS" if result.status == PipelineStatus.COMPLETED else "❌ FAILED"
            self.logger.info(f"{result.name:20} | {status:10} | {result.execution_time:8.2f}s")
            
            if result.error:
                self.logger.info(f"{'':20} | Error: {result.error}")
            
            if result.status == PipelineStatus.COMPLETED:
                successful_pipelines += 1
            else:
                failed_pipelines += 1
        
        self.logger.info("-" * 100)
        self.logger.info(f"Total Execution Time: {total_time:.2f} seconds")
        self.logger.info(f"Successful Pipelines: {successful_pipelines}/{len(self.results)}")
        self.logger.info(f"Failed Pipelines: {failed_pipelines}/{len(self.results)}")
        
        # Validation summary
        validation_summary = validator_orchestrator.get_validation_summary()
        self.logger.info(f"Total Validations: {validation_summary.get('total_validations', 0)}")
        self.logger.info(f"Validation Success Rate: {validation_summary.get('success_rate', 0):.2%}")
        
        if failed_pipelines == 0:
            self.logger.info("🎉 ALL PIPELINES COMPLETED SUCCESSFULLY!")
        else:
            self.logger.info(f"⚠️  {failed_pipelines} PIPELINE(S) FAILED")
        
        self.logger.info("=" * 100)
        
        # Save comprehensive results
        await self._save_comprehensive_results(total_time, config)
    
    @handles_errors(Exception, fallback=None)
    async def _save_comprehensive_results(self, total_time: float, config: Dict[str, Any]) -> None:
        """Save comprehensive results including validation and performance data."""
        
        results_file = Path(self.data_dir) / f"enhanced_pipeline_results_{self.symbol}_{self.timeframe}.json"
        
        comprehensive_results = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'data_dir': self.data_dir,
            'total_execution_time': total_time,
            'successful_pipelines': sum(1 for r in self.results if r.status == PipelineStatus.COMPLETED),
            'failed_pipelines': sum(1 for r in self.results if r.status == PipelineStatus.FAILED),
            'config': config,
            'pipeline_results': [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "error": result.error,
                    "validation_results": result.validation_results,
                    "performance_metrics": result.performance_metrics,
                    "data_quality_metrics": result.data_quality_metrics
                }
                for result in self.results
            ],
            'validation_summary': validator_orchestrator.get_validation_summary(),
            'pipeline_utilities_status': pipeline_utilities.get_pipeline_status(),
            'timestamp': format_datetime(get_current_datetime())
        }
        
        ensure_directory(Path(results_file).parent)
        safe_json_dump(comprehensive_results, results_file, indent=2)
        self.logger.info(f"💾 Comprehensive results saved to: {results_file}")
        
        # Save validation report
        validation_report_file = Path(self.data_dir) / f"validation_report_{self.symbol}_{self.timeframe}.json"
        validator_orchestrator.save_validation_report(validation_report_file)
    
    def _all_pipelines_successful(self) -> bool:
        """Check if all pipelines completed successfully."""
        return all(result.status == PipelineStatus.COMPLETED for result in self.results)


async def run_all_pipelines(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **config: Dict[str, Any]
) -> bool:
    """Run all training pipelines with enhanced validation and monitoring."""
    
    orchestrator = EnhancedPipelineOrchestrator(symbol, exchange, timeframe, data_dir)
    return await orchestrator.run_all_pipelines(**config)

async def main():
    """Main function to run all pipelines."""
    
    # Default configuration
    config = {
        'force_rerun': True,
        'quality_checks': True,
        'validate_data': True,
        'convert_format': True,
        'hmm_clustering': True,
        'regime_splitting': True,
        'feature_engineering': True,
        'matrix_operations': True,
        'feature_selection': True,
        'hmm_training': True,
        'regime_intelligence': True,
        'analyst_creation': True,
        'analyst_enhancement': True,
        'ensemble_creation': True,
        'tactician_training': True,
        'confidence_calibration': True,
        'parameter_optimization': True,
        'walk_forward_validation': True,
        'monte_carlo_validation': True,
        'ab_testing': True,
        'model_saving': True,
        'random_state': 42,
    }
    
    success = await run_all_pipelines(**config)
    
    if success:
        print("\n🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
        sys.exit(0)
    else:
        print("\n❌ PIPELINE EXECUTION FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    # Run all pipelines
    asyncio.run(main())