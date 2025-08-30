#!/usr/bin/env python3
"""Step 2: Data Reading and Validation.

This module handles reading the unified data from step1_5 and performs comprehensive
data quality validation before proceeding to HMM regime discovery.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

# Handle optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
    monitor_feature_engineering,
)
from src.utils.logger import system_logger

logger = system_logger.getChild("Step2DataReading")


class DataReadingStep:
    """Step 2: Data Reading and Validation with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("DataReadingStep")
        self.data_quality_manager = None
        self.start_time = None
        self.step_timings = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize data reading and quality components."""
        self.logger.info("🔧 Initializing data reading components...")
        try:
            from .step1.enhanced_data_quality_manager import EnhancedDataQualityManager
            self.data_quality_manager = EnhancedDataQualityManager()
            self.logger.info("✅ Enhanced data quality manager initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import EnhancedDataQualityManager: {e}")
            self.logger.info("📝 Proceeding without enhanced data quality manager")

    async def initialize(self) -> None:
        """Initialize the data reading step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing Data Reading Step...")
        self.logger.info("📋 Step 2 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info("✅ Data Reading Step initialized successfully")

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("read_unified_data")
    @quality_gate(
        min_quality_score=0.8,
        max_correlation=0.95,
        required_grade="B"
    )
    @comprehensive_data_validation
    @memory_efficient
    async def read_unified_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Read unified data from step1_5 output."""
        step_start = time.time()
        self.logger.info(f"📖 Reading unified data for {symbol} on {exchange} ({timeframe})")
        
        try:
            # Construct path to unified data in structured directory
            unified_data_path = Path(data_dir) / "unified" / exchange.lower() / symbol / timeframe
            
            if not unified_data_path.exists():
                self.logger.error(f"❌ Unified data path does not exist: {unified_data_path}")
                return None
            
            # Find all parquet files in the directory
            parquet_files = list(unified_data_path.glob("*.parquet"))
            
            if not parquet_files:
                self.logger.error(f"❌ No parquet files found in {unified_data_path}")
                return None
            
            self.logger.info(f"📁 Found {len(parquet_files)} parquet files")
            
            # Read and concatenate all parquet files
            dataframes = []
            for file_path in sorted(parquet_files):
                self.logger.info(f"📖 Reading {file_path.name}")
                df = pd.read_parquet(file_path)
                dataframes.append(df)
            
            # Concatenate all dataframes
            if dataframes:
                unified_data = pd.concat(dataframes, ignore_index=True)
                unified_data = unified_data.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"✅ Successfully read unified data: {len(unified_data)} rows")
                self._log_step_timing("read_unified_data", step_start)
                
                return unified_data
            else:
                self.logger.error("❌ No data found in parquet files")
                return None
                
        except Exception as e:
            self.logger.exception(f"❌ Error reading unified data: {e}")
            return None

    @with_tracing_span("validate_data_quality")
    @comprehensive_data_validation
    async def validate_data_quality(self, data: pd.DataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """Validate data quality and structure."""
        step_start = time.time()
        self.logger.info("🔍 Validating data quality...")
        
        validation_results = {
            "passed": True,
            "issues": [],
            "warnings": [],
            "data_info": {}
        }
        
        try:
            # Basic data structure validation
            if data is None or data.empty:
                validation_results["passed"] = False
                validation_results["issues"].append("Data is None or empty")
                return validation_results
            
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_results["passed"] = False
                validation_results["issues"].append(f"Missing required columns: {missing_columns}")
            
            # Check for null values
            null_counts = data[required_columns].isnull().sum()
            if null_counts.sum() > 0:
                validation_results["warnings"].append(f"Found null values: {null_counts.to_dict()}")
            
            # Check data types
            expected_types = {
                'timestamp': 'datetime64[ns]',
                'open': 'float64',
                'high': 'float64', 
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            }
            
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        validation_results["warnings"].append(f"Column {col} has type {actual_type}, expected {expected_type}")
            
            # Check for duplicate timestamps
            if 'timestamp' in data.columns:
                duplicates = data['timestamp'].duplicated().sum()
                if duplicates > 0:
                    validation_results["warnings"].append(f"Found {duplicates} duplicate timestamps")
            
            # Check for price anomalies
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Check for negative prices
                price_columns = ['open', 'high', 'low', 'close']
                negative_prices = (data[price_columns] < 0).any(axis=1).sum()
                if negative_prices > 0:
                    validation_results["issues"].append(f"Found {negative_prices} rows with negative prices")
                
                # Check for high-low consistency
                invalid_hl = ((data['high'] < data['low']) | 
                             (data['open'] > data['high']) | 
                             (data['close'] > data['high']) |
                             (data['open'] < data['low']) | 
                             (data['close'] < data['low'])).sum()
                
                if invalid_hl > 0:
                    validation_results["issues"].append(f"Found {invalid_hl} rows with invalid OHLC relationships")
            
            # Store data information
            validation_results["data_info"] = {
                "rows": len(data),
                "columns": list(data.columns),
                "date_range": {
                    "start": data['timestamp'].min() if 'timestamp' in data.columns else None,
                    "end": data['timestamp'].max() if 'timestamp' in data.columns else None
                },
                "memory_usage": data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            self.logger.info(f"✅ Data quality validation completed")
            self.logger.info(f"   - Rows: {validation_results['data_info']['rows']}")
            self.logger.info(f"   - Memory usage: {validation_results['data_info']['memory_usage']:.2f} MB")
            self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
            self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")
            
            self._log_step_timing("validate_data_quality", step_start)
            
        except Exception as e:
            self.logger.exception(f"❌ Error during data quality validation: {e}")
            validation_results["passed"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
        
        return validation_results

    @with_tracing_span("save_validation_report")
    async def save_validation_report(self, validation_results: Dict[str, Any], symbol: str, exchange: str, data_dir: str) -> bool:
        """Save validation report to file."""
        step_start = time.time()
        self.logger.info("💾 Saving validation report...")
        
        try:
            import json
            from datetime import datetime
            
            # Create reports directory
            reports_dir = Path(data_dir) / "reports" / "data_quality"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Create report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"data_reading_validation_{exchange}_{symbol}_{timestamp}.json"
            report_path = reports_dir / report_filename
            
            # Prepare report data
            report_data = {
                "step": "step2_data_reading",
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "exchange": exchange,
                "validation_results": validation_results,
                "step_timings": self.step_timings
            }
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Validation report saved to {report_path}")
            self._log_step_timing("save_validation_report", step_start)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving validation report: {e}")
            return False

    @with_tracing_span("execute_data_reading_step")
    @handle_errors
    @resource_monitor
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute the complete data reading step."""
        self.logger.info("🚀 Starting Step 2: Data Reading and Validation")
        
        try:
            # Read unified data
            unified_data = await self.read_unified_data(symbol, exchange, timeframe, data_dir)
            
            if unified_data is None:
                self.logger.error("❌ Failed to read unified data")
                return {"success": False, "error": "Failed to read unified data"}
            
            # Validate data quality
            validation_results = await self.validate_data_quality(unified_data, symbol, exchange)
            
            # Save validation report
            await self.save_validation_report(validation_results, symbol, exchange, data_dir)
            
            # Check if validation passed
            if not validation_results["passed"]:
                self.logger.error("❌ Data quality validation failed")
                self.logger.error(f"   Issues: {validation_results['issues']}")
                return {
                    "success": False, 
                    "error": "Data quality validation failed",
                    "validation_results": validation_results
                }
            
            # Save processed data for next step
            output_path = Path(data_dir) / "processed" / f"{exchange}_{symbol}_{timeframe}_validated_data.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            unified_data.to_parquet(output_path, index=False)
            
            self.logger.info(f"✅ Step 2 completed successfully")
            self.logger.info(f"   - Validated data saved to: {output_path}")
            self.logger.info(f"   - Total execution time: {time.time() - self.start_time:.2f} seconds")
            
            return {
                "success": True,
                "data_path": str(output_path),
                "validation_results": validation_results,
                "step_timings": self.step_timings
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Error in Step 2: {e}")
            return {"success": False, "error": str(e)}


async def run_step_enhanced(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,  # Will be constructed as data_cache/exchange/asset/
    **kwargs
) -> Dict[str, Any]:
    """Enhanced entry point for Step 2: Data Reading and Validation."""
    
    # Construct structured data directory
    if data_dir is None:
        data_dir = os.path.join("data_cache", exchange.lower(), symbol.lower())
    
    logger.info("🚀 Starting Step 2: Data Reading and Validation (Enhanced)")
    
    # Create configuration
    config = {
        "SYMBOL": symbol,
        "EXCHANGE": exchange,
        "TIMEFRAME": timeframe,
        "DATA_DIR": data_dir,
        **kwargs
    }
    
    # Initialize step
    step = DataReadingStep(config)
    await step.initialize()
    
    # Execute step
    result = await step.execute(symbol, exchange, timeframe, data_dir, **kwargs)
    
    if result["success"]:
        logger.info("✅ Step 2: Data Reading and Validation completed successfully")
    else:
        logger.error(f"❌ Step 2: Data Reading and Validation failed: {result.get('error', 'Unknown error')}")
    
    return result


async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,  # Will be constructed as data_cache/exchange/asset/
    **kwargs
) -> bool:
    """Standard entry point for Step 2: Data Reading and Validation."""
    
    result = await run_step_enhanced(symbol, exchange, timeframe, data_dir, **kwargs)
    return result["success"]


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step_enhanced(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir=None  # Will use structured directory
        )
        print(f"Result: {result}")
    
    asyncio.run(test())