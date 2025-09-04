#!/usr/bin/env python3
"""
Enhanced Data Collection Pipeline

This module provides a comprehensive data collection pipeline with:
- Comprehensive validators at each step
- Decorators for data protection
- Common utilities for data operations
- Enhanced error handling
- Real-time monitoring
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Import our enhanced components
from .validators.pipeline_validators import (
    DataCollectionValidator,
    ValidationResult,
    ValidationReport
)
from .decorators.step_decorators import (
    data_operation_protection,
    data_formatting_protection,
    data_analysis_protection,
    step_execution_protection,
    DataOperationType,
    SecurityLevel
)
from .utils.data_operations_utils import (
    DataFormatter,
    DataAnalyzer,
    DataAccessManager,
    DataStorageManager,
    ErrorHandler,
    DataFormat,
    CompressionType
)
from .error_handling.enhanced_error_handler import (
    EnhancedErrorHandler,
    ErrorContext,
    DataQualityError,
    NetworkError,
    StorageError,
    ValidationError,
    ProcessingError
)
from .monitoring.pipeline_monitor import (
    PipelineMonitor,
    StepMonitor,
    MonitorStatus
)

from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    ensure_directory
)


class EnhancedDataCollectionPipeline:
    """Enhanced data collection pipeline with comprehensive protection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validator = DataCollectionValidator(config)
        self.data_formatter = DataFormatter(config)
        self.data_analyzer = DataAnalyzer(config)
        self.data_access_manager = DataAccessManager(config)
        self.data_storage_manager = DataStorageManager(config)
        self.error_handler = ErrorHandler(config)
        self.enhanced_error_handler = EnhancedErrorHandler(config)
        
        # Initialize monitoring
        self.pipeline_monitor: Optional[PipelineMonitor] = None
        self.step_monitors: Dict[str, StepMonitor] = {}
        
        # Pipeline state
        self.pipeline_id = f"data_collection_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}"
        self.symbol: Optional[str] = None
        self.exchange: Optional[str] = None
        self.data_dir: Optional[str] = None
    
    @step_execution_protection(
        step_name="data_collection_pipeline",
        prerequisites=[],
        outputs=["data_collected", "data_validated", "data_formatted"],
        quality_threshold=0.8
    )
    async def run_pipeline(
        self,
        symbol: str,
        exchange: str,
        data_dir: str = "data_cache",
        **kwargs
    ) -> Dict[str, Any]:
        """Run the enhanced data collection pipeline."""
        try:
            # Initialize pipeline
            self.symbol = symbol
            self.exchange = exchange
            self.data_dir = data_dir
            
            # Start monitoring
            self.pipeline_monitor = PipelineMonitor(self.pipeline_id, self.config)
            self.pipeline_monitor.start_pipeline(total_steps=3)
            
            self.logger.info(f"🚀 Starting enhanced data collection pipeline for {symbol} on {exchange}")
            print(f"🚀 Starting enhanced data collection pipeline for {symbol} on {exchange}")
            print("="*80)
            
            # Step 1: Data Collection
            step1_result = await self._run_step1_data_collection()
            if not step1_result.get("success", False):
                await self._handle_pipeline_failure("Step 1: Data Collection failed")
                return step1_result
            
            # Step 2: Data Validation
            step2_result = await self._run_step2_data_validation()
            if not step2_result.get("success", False):
                await self._handle_pipeline_failure("Step 2: Data Validation failed")
                return step2_result
            
            # Step 3: Data Formatting and Storage
            step3_result = await self._run_step3_data_formatting()
            if not step3_result.get("success", False):
                await self._handle_pipeline_failure("Step 3: Data Formatting failed")
                return step3_result
            
            # Complete pipeline
            await self._complete_pipeline()
            
            # Generate final report
            final_report = await self._generate_final_report()
            
            self.logger.info("✅ Enhanced data collection pipeline completed successfully")
            print("✅ Enhanced data collection pipeline completed successfully")
            
            return final_report
            
        except Exception as e:
            self.logger.exception(f"Pipeline execution failed: {e}")
            await self._handle_pipeline_failure(f"Pipeline execution failed: {e}")
            raise
    
    @data_operation_protection(
        operation_type=DataOperationType.READ,
        security_level=SecurityLevel.INTERNAL,
        audit=True,
        validate_inputs=True,
        validate_outputs=True,
        timeout_seconds=300,
        retry_attempts=3
    )
    async def _run_step1_data_collection(self) -> Dict[str, Any]:
        """Run Step 1: Data Collection with protection."""
        step_name = "step1_data_collection"
        step_monitor = self.pipeline_monitor.start_step(step_name)
        
        try:
            self.logger.info(f"📊 Running {step_name}")
            print(f"📊 Running {step_name}")
            
            # Create error context
            error_context = ErrorContext(
                operation="data_collection",
                step_name=step_name,
                symbol=self.symbol,
                exchange=self.exchange,
                data_dir=self.data_dir
            )
            
            # Execute data collection with error handling
            result = await self.enhanced_error_handler.execute_with_error_handling(
                self._collect_raw_data,
                error_context,
                self.symbol,
                self.exchange,
                self.data_dir
            )
            
            if result is None:
                raise DataQualityError("Data collection returned no results", error_context)
            
            # Record metrics
            step_monitor.record_data_processed(len(result) if isinstance(result, pd.DataFrame) else 0)
            step_monitor.set_custom_metric("data_rows", len(result) if isinstance(result, pd.DataFrame) else 0)
            
            self.pipeline_monitor.end_step(step_name, MonitorStatus.COMPLETED)
            
            return {
                "success": True,
                "step": step_name,
                "data": result,
                "message": f"{step_name} completed successfully"
            }
            
        except Exception as e:
            self.pipeline_monitor.end_step(step_name, MonitorStatus.FAILED)
            step_monitor.record_error()
            raise
    
    @data_operation_protection(
        operation_type=DataOperationType.VALIDATE,
        security_level=SecurityLevel.INTERNAL,
        audit=True,
        validate_inputs=True,
        validate_outputs=True
    )
    async def _run_step2_data_validation(self) -> Dict[str, Any]:
        """Run Step 2: Data Validation with protection."""
        step_name = "step2_data_validation"
        step_monitor = self.pipeline_monitor.start_step(step_name)
        
        try:
            self.logger.info(f"🔍 Running {step_name}")
            print(f"🔍 Running {step2_data_validation}")
            
            # Create error context
            error_context = ErrorContext(
                operation="data_validation",
                step_name=step_name,
                symbol=self.symbol,
                exchange=self.exchange,
                data_dir=self.data_dir
            )
            
            # Execute validation with error handling
            validation_result = await self.enhanced_error_handler.execute_with_error_handling(
                self._validate_data_quality,
                error_context,
                self.symbol,
                self.exchange,
                self.data_dir
            )
            
            if validation_result is None:
                raise ValidationError("Data validation returned no results", error_context)
            
            # Check validation results
            if validation_result.result == ValidationResult.FAILED:
                raise DataQualityError(
                    f"Data validation failed: {validation_result.message}",
                    error_context
                )
            
            # Record metrics
            step_monitor.set_custom_metric("validation_result", validation_result.result.value)
            step_monitor.set_custom_metric("warnings", len(validation_result.warnings))
            step_monitor.set_custom_metric("errors", len(validation_result.errors))
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    step_monitor.record_warning()
            
            self.pipeline_monitor.end_step(step_name, MonitorStatus.COMPLETED)
            
            return {
                "success": True,
                "step": step_name,
                "validation_result": validation_result,
                "message": f"{step_name} completed successfully"
            }
            
        except Exception as e:
            self.pipeline_monitor.end_step(step_name, MonitorStatus.FAILED)
            step_monitor.record_error()
            raise
    
    @data_formatting_protection(
        required_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        data_types={'timestamp': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'},
        min_rows=100,
        max_null_ratio=0.05,
        check_duplicates=True,
        check_timestamps=True
    )
    async def _run_step3_data_formatting(self) -> Dict[str, Any]:
        """Run Step 3: Data Formatting and Storage with protection."""
        step_name = "step3_data_formatting"
        step_monitor = self.pipeline_monitor.start_step(step_name)
        
        try:
            self.logger.info(f"🔄 Running {step_name}")
            print(f"🔄 Running {step_name}")
            
            # Create error context
            error_context = ErrorContext(
                operation="data_formatting",
                step_name=step_name,
                symbol=self.symbol,
                exchange=self.exchange,
                data_dir=self.data_dir
            )
            
            # Execute formatting with error handling
            formatted_data = await self.enhanced_error_handler.execute_with_error_handling(
                self._format_and_store_data,
                error_context,
                self.symbol,
                self.exchange,
                self.data_dir
            )
            
            if formatted_data is None:
                raise ProcessingError("Data formatting returned no results", error_context)
            
            # Record metrics
            step_monitor.record_data_processed(len(formatted_data) if isinstance(formatted_data, pd.DataFrame) else 0)
            step_monitor.set_custom_metric("formatted_rows", len(formatted_data) if isinstance(formatted_data, pd.DataFrame) else 0)
            
            self.pipeline_monitor.end_step(step_name, MonitorStatus.COMPLETED)
            
            return {
                "success": True,
                "step": step_name,
                "formatted_data": formatted_data,
                "message": f"{step_name} completed successfully"
            }
            
        except Exception as e:
            self.pipeline_monitor.end_step(step_name, MonitorStatus.FAILED)
            step_monitor.record_error()
            raise
    
    async def _collect_raw_data(
        self,
        symbol: str,
        exchange: str,
        data_dir: str
    ) -> pd.DataFrame:
        """Collect raw data from exchange."""
        # This would implement the actual data collection logic
        # For now, return a placeholder DataFrame
        import numpy as np
        
        # Create sample data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(150, 250, 1000),
            'low': np.random.uniform(50, 150, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        self.logger.info(f"Collected {len(df)} rows of raw data for {symbol} on {exchange}")
        return df
    
    async def _validate_data_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str
    ) -> ValidationReport:
        """Validate data quality."""
        # Run comprehensive validation
        validation_result = await self.validator.validate_step1_data_collection(
            symbol, exchange, data_dir
        )
        
        self.logger.info(f"Data validation completed: {validation_result.result.value}")
        return validation_result
    
    async def _format_and_store_data(
        self,
        symbol: str,
        exchange: str,
        data_dir: str
    ) -> pd.DataFrame:
        """Format and store data."""
        # Get the collected data (in a real implementation, this would come from step 1)
        raw_data = await self._collect_raw_data(symbol, exchange, data_dir)
        
        # Format the data
        formatting_result = self.data_formatter.format_klines_data(
            raw_data, symbol, exchange
        )
        
        if not formatting_result.success:
            raise ProcessingError(f"Data formatting failed: {formatting_result.message}")
        
        formatted_data = formatting_result.data
        
        # Analyze data quality
        quality_metrics = self.data_analyzer.analyze_data_quality(formatted_data, "klines")
        
        # Store the data
        output_file = Path(data_dir) / f"formatted_{exchange}_{symbol}_klines.parquet"
        storage_result = self.data_storage_manager.save_data(
            formatted_data,
            output_file,
            format=DataFormat.PARQUET,
            compression=CompressionType.GZIP,
            metadata={
                "symbol": symbol,
                "exchange": exchange,
                "quality_metrics": quality_metrics.__dict__,
                "pipeline_id": self.pipeline_id,
                "created_at": format_datetime(get_current_datetime())
            }
        )
        
        if not storage_result.success:
            raise StorageError(f"Data storage failed: {storage_result.message}")
        
        self.logger.info(f"Data formatted and stored successfully: {output_file}")
        return formatted_data
    
    async def _handle_pipeline_failure(self, error_message: str) -> None:
        """Handle pipeline failure."""
        self.logger.error(f"Pipeline failure: {error_message}")
        print(f"❌ Pipeline failure: {error_message}")
        
        if self.pipeline_monitor:
            self.pipeline_monitor.end_pipeline(MonitorStatus.FAILED)
            self.pipeline_monitor.print_progress_report()
    
    async def _complete_pipeline(self) -> None:
        """Complete the pipeline successfully."""
        if self.pipeline_monitor:
            self.pipeline_monitor.end_pipeline(MonitorStatus.COMPLETED)
            self.pipeline_monitor.print_progress_report()
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final pipeline report."""
        pipeline_metrics = self.pipeline_monitor.get_pipeline_metrics() if self.pipeline_monitor else None
        
        # Get validation summary
        validation_summary = self.validator.get_validation_summary()
        
        # Get error summary
        error_summary = self.enhanced_error_handler.get_error_summary()
        
        report = {
            "pipeline_id": self.pipeline_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "data_dir": self.data_dir,
            "status": "COMPLETED",
            "timestamp": format_datetime(get_current_datetime()),
            "pipeline_metrics": pipeline_metrics.__dict__ if pipeline_metrics else None,
            "validation_summary": validation_summary,
            "error_summary": error_summary,
            "success": True
        }
        
        return report


# Main execution function
async def run_enhanced_data_collection_pipeline(
    symbol: str,
    exchange: str,
    data_dir: str = "data_cache",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the enhanced data collection pipeline."""
    if config is None:
        config = {
            "max_retry_attempts": 3,
            "retry_delays": [1, 2, 4],
            "error_log_file": "logs/error_reports.json",
            "monitoring_file": "logs/pipeline_monitor.json"
        }
    
    # Ensure data directory exists
    ensure_directory(data_dir)
    
    # Create and run pipeline
    pipeline = EnhancedDataCollectionPipeline(config)
    result = await pipeline.run_pipeline(symbol, exchange, data_dir)
    
    return result


if __name__ == "__main__":
    # Example usage
    async def main():
        result = await run_enhanced_data_collection_pipeline(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_dir="data_cache"
        )
        print(f"Pipeline result: {result}")
    
    asyncio.run(main())