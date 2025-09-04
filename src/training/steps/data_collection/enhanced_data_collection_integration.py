#!/usr/bin/env python3
"""
Enhanced Data Collection Integration

This module demonstrates how to integrate the enhanced data validation framework
into the existing data collection pipeline for Step01 and Step01_5.

Features:
- Real-time schema enforcement during API collection
- Comprehensive data quality validation
- Time gap detection between batches
- Field mapping for different exchanges
- Integration with existing pipeline
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from .enhanced_data_validation_framework import DataType, ValidationSeverity
from .enhanced_data_collector import EnhancedDataCollectionManager, collect_all_data_with_validation
from .enhanced_step01_data_collection import run_enhanced_step01_data_collection
from .enhanced_step01_5_data_converter import run_enhanced_step01_5_data_converter

logger = system_logger.getChild("EnhancedDataCollectionIntegration")


class EnhancedDataCollectionPipeline:
    """Enhanced data collection pipeline with comprehensive validation."""
    
    def __init__(self, exchange: str, symbol: str, timeframe: str):
        self.exchange = exchange.upper()
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger.getChild(f"Pipeline.{exchange}.{symbol}")
        
        # Pipeline state
        self.pipeline_start_time = time.time()
        self.pipeline_state = {
            'step01_completed': False,
            'step01_5_completed': False,
            'overall_success': False,
            'validation_summary': {},
            'quality_metrics': {}
        }
    
    async def run_complete_pipeline(self, force_rerun: bool = False) -> Dict[str, Any]:
        """Run the complete enhanced data collection pipeline."""
        
        self.logger.info("🚀 Starting Enhanced Data Collection Pipeline")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 Symbol: {self.symbol}")
        self.logger.info(f"🏢 Exchange: {self.exchange}")
        self.logger.info(f"📊 Timeframe: {self.timeframe}")
        self.logger.info(f"🔄 Force rerun: {force_rerun}")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Enhanced Data Collection
            self.logger.info("📥 STEP 1: Enhanced Data Collection")
            self.logger.info("-" * 40)
            
            step01_success = await run_enhanced_step01_data_collection(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                force_rerun=force_rerun
            )
            
            self.pipeline_state['step01_completed'] = step01_success
            
            if not step01_success:
                self.logger.error("❌ Step 1 failed - stopping pipeline")
                return self._create_pipeline_summary()
            
            self.logger.info("✅ Step 1 completed successfully")
            
            # Step 1.5: Enhanced Data Converter
            self.logger.info("🔄 STEP 1.5: Enhanced Data Converter")
            self.logger.info("-" * 40)
            
            step01_5_success = await run_enhanced_step01_5_data_converter(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                force_rerun=force_rerun
            )
            
            self.pipeline_state['step01_5_completed'] = step01_5_success
            
            if not step01_5_success:
                self.logger.error("❌ Step 1.5 failed - stopping pipeline")
                return self._create_pipeline_summary()
            
            self.logger.info("✅ Step 1.5 completed successfully")
            
            # Pipeline completed successfully
            self.pipeline_state['overall_success'] = True
            
            # Generate final summary
            summary = self._create_pipeline_summary()
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 ENHANCED DATA COLLECTION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"📊 Total Duration: {summary['total_duration']:.2f} seconds")
            self.logger.info(f"✅ Overall Success: {summary['overall_success']}")
            self.logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline failed with exception: {e}")
            return self._create_pipeline_summary()
    
    def _create_pipeline_summary(self) -> Dict[str, Any]:
        """Create comprehensive pipeline summary."""
        total_duration = time.time() - self.pipeline_start_time
        
        return {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_duration': total_duration,
            'overall_success': self.pipeline_state['overall_success'],
            'step01_completed': self.pipeline_state['step01_completed'],
            'step01_5_completed': self.pipeline_state['step01_5_completed'],
            'pipeline_state': self.pipeline_state,
            'timestamp': datetime.now().isoformat()
        }


async def run_enhanced_data_collection_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    force_rerun: bool = False
) -> Dict[str, Any]:
    """
    Run the complete enhanced data collection pipeline.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'BINANCE')
        timeframe: Timeframe (e.g., '1m')
        force_rerun: Whether to force rerun even if data exists
        
    Returns:
        Pipeline execution summary
    """
    
    pipeline = EnhancedDataCollectionPipeline(exchange, symbol, timeframe)
    return await pipeline.run_complete_pipeline(force_rerun)


async def validate_existing_data(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache"
) -> Dict[str, Any]:
    """
    Validate existing data files against enhanced schemas.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        
    Returns:
        Validation summary
    """
    
    logger.info(f"🔍 Validating existing data for {exchange}_{symbol}_{timeframe}")
    
    try:
        from .enhanced_data_validation_framework import get_validator
        import pandas as pd
        import os
        
        validation_results = {}
        
        # Validate klines data
        klines_file = f"klines_{exchange}_{symbol}_{timeframe}_validated.parquet"
        klines_path = os.path.join(data_dir, klines_file)
        
        if os.path.exists(klines_path):
            logger.info(f"📖 Validating klines data: {klines_file}")
            df = pd.read_parquet(klines_path)
            
            # Convert to validation format
            rows = df.to_dict('records')
            
            # Validate with enhanced validator
            validator = get_validator(DataType.KLINES)
            validated_rows = []
            
            for i, row in enumerate(rows):
                try:
                    validated_row = validator.validate_row(row, i)
                    validated_rows.append(validated_row)
                except Exception as e:
                    logger.warning(f"⚠️ Row {i} validation failed: {e}")
            
            validation_results['klines'] = {
                'total_rows': len(rows),
                'valid_rows': len(validated_rows),
                'success_rate': len(validated_rows) / len(rows) * 100 if rows else 0,
                'validation_summary': validator.get_validation_summary()
            }
            
            logger.info(f"✅ Klines validation: {len(validated_rows)}/{len(rows)} rows valid")
        
        # Validate aggtrades data
        aggtrades_file = f"aggtrades_{exchange}_{symbol}_validated.parquet"
        aggtrades_path = os.path.join(data_dir, aggtrades_file)
        
        if os.path.exists(aggtrades_path):
            logger.info(f"📖 Validating aggtrades data: {aggtrades_file}")
            df = pd.read_parquet(aggtrades_path)
            
            # Convert to validation format
            rows = df.to_dict('records')
            
            # Validate with enhanced validator
            validator = get_validator(DataType.AGGTRADES)
            validated_rows = []
            
            for i, row in enumerate(rows):
                try:
                    validated_row = validator.validate_row(row, i)
                    validated_rows.append(validated_row)
                except Exception as e:
                    logger.warning(f"⚠️ Row {i} validation failed: {e}")
            
            validation_results['aggtrades'] = {
                'total_rows': len(rows),
                'valid_rows': len(validated_rows),
                'success_rate': len(validated_rows) / len(rows) * 100 if rows else 0,
                'validation_summary': validator.get_validation_summary()
            }
            
            logger.info(f"✅ Aggtrades validation: {len(validated_rows)}/{len(rows)} rows valid")
        
        # Validate futures data
        futures_file = f"futures_{exchange}_{symbol}_validated.parquet"
        futures_path = os.path.join(data_dir, futures_file)
        
        if os.path.exists(futures_path):
            logger.info(f"📖 Validating futures data: {futures_file}")
            df = pd.read_parquet(futures_path)
            
            # Convert to validation format
            rows = df.to_dict('records')
            
            # Validate with enhanced validator
            validator = get_validator(DataType.FUTURES)
            validated_rows = []
            
            for i, row in enumerate(rows):
                try:
                    validated_row = validator.validate_row(row, i)
                    validated_rows.append(validated_row)
                except Exception as e:
                    logger.warning(f"⚠️ Row {i} validation failed: {e}")
            
            validation_results['futures'] = {
                'total_rows': len(rows),
                'valid_rows': len(validated_rows),
                'success_rate': len(validated_rows) / len(rows) * 100 if rows else 0,
                'validation_summary': validator.get_validation_summary()
            }
            
            logger.info(f"✅ Futures validation: {len(validated_rows)}/{len(rows)} rows valid")
        
        # Calculate overall validation summary
        total_rows = sum(result['total_rows'] for result in validation_results.values())
        total_valid_rows = sum(result['valid_rows'] for result in validation_results.values())
        overall_success_rate = total_valid_rows / total_rows * 100 if total_rows > 0 else 0
        
        validation_summary = {
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'total_rows': total_rows,
            'total_valid_rows': total_valid_rows,
            'overall_success_rate': overall_success_rate,
            'validation_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("=" * 60)
        logger.info("📊 DATA VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🎯 Symbol: {symbol}")
        logger.info(f"🏢 Exchange: {exchange}")
        logger.info(f"📊 Timeframe: {timeframe}")
        logger.info(f"📊 Total Rows: {total_rows}")
        logger.info(f"✅ Valid Rows: {total_valid_rows}")
        logger.info(f"📈 Overall Success Rate: {overall_success_rate:.1f}%")
        logger.info("=" * 60)
        
        return validation_summary
        
    except Exception as e:
        logger.exception(f"❌ Error validating existing data: {e}")
        return {
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


async def demonstrate_enhanced_validation():
    """Demonstrate the enhanced validation framework."""
    
    logger.info("🎯 Demonstrating Enhanced Data Validation Framework")
    logger.info("=" * 80)
    
    # Example 1: Validate existing data
    logger.info("📋 Example 1: Validating existing data")
    validation_summary = await validate_existing_data(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m"
    )
    
    logger.info(f"Validation Summary: {validation_summary}")
    
    # Example 2: Run complete pipeline
    logger.info("📋 Example 2: Running complete enhanced pipeline")
    pipeline_summary = await run_enhanced_data_collection_pipeline(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=False
    )
    
    logger.info(f"Pipeline Summary: {pipeline_summary}")
    
    # Example 3: Test with different data types
    logger.info("📋 Example 3: Testing different data types")
    
    from .enhanced_data_validation_framework import get_validator
    
    # Test klines validation
    klines_validator = get_validator(DataType.KLINES)
    test_klines_data = [
        {
            "open_time": 1640995200000,
            "open": "3000.0",
            "high": "3100.0",
            "low": "2900.0",
            "close": "3050.0",
            "volume": "1000.0"
        }
    ]
    
    validated_klines = klines_validator.validate_batch(test_klines_data)
    logger.info(f"✅ Validated {len(validated_klines)} klines rows")
    
    # Test aggtrades validation
    aggtrades_validator = get_validator(DataType.AGGTRADES)
    test_aggtrades_data = [
        {
            "T": 1640995200000,
            "p": "3050.0",
            "q": "1.5",
            "m": True
        }
    ]
    
    validated_aggtrades = aggtrades_validator.validate_batch(test_aggtrades_data)
    logger.info(f"✅ Validated {len(validated_aggtrades)} aggtrades rows")
    
    # Test futures validation
    futures_validator = get_validator(DataType.FUTURES)
    test_futures_data = [
        {
            "fundingTime": 1640995200000,
            "fundingRate": "0.0001"
        }
    ]
    
    validated_futures = futures_validator.validate_batch(test_futures_data)
    logger.info(f"✅ Validated {len(validated_futures)} futures rows")
    
    logger.info("=" * 80)
    logger.info("🎉 Enhanced validation framework demonstration completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_enhanced_validation())