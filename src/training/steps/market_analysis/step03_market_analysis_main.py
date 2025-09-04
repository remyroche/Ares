#!/usr/bin/env python3
"""Step 3: Enhanced Market Analysis Pipeline.

This module provides the main interface for enhanced market analysis with:
1. Comprehensive validation at each step
2. Data protection and security
3. Error handling and recovery
4. Performance monitoring
5. Step orchestration with proper flow control
6. Comprehensive validation framework
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.enhanced_market_analysis_pipeline import run_enhanced_market_analysis_pipeline
from src.training.steps.market_analysis.step_orchestrator import run_market_analysis_orchestrator
from src.utils.comprehensive_validation_framework import (
    comprehensive_validation_framework,
    ValidationLevel,
    validate_pipeline_comprehensive
)
from src.utils.logger import system_logger

logger = system_logger.getChild("MarketAnalysisMain")

async def main():
    """Main function to run enhanced market analysis pipeline."""
    print("🚀 Step 3: Enhanced Market Analysis Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Enhanced market analysis parameters
    config = {
        'force_rerun': True,
        'enable_data_collection': True,
        'enable_hmm_clustering': True,
        'enable_feature_engineering': True,
        'validation_level': ValidationLevel.COMPREHENSIVE,
        'data_protection': True,
        'performance_monitoring': True,
        'random_state': 42,
    }
    
    print(f"📊 Enhanced Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Validation level: {config['validation_level'].value}")
    print(f"   Data protection: {config['data_protection']}")
    print(f"   Performance monitoring: {config['performance_monitoring']}")
    print("=" * 80)
    
    # Initialize validation framework
    print("🔧 Initializing comprehensive validation framework...")
    validation_init_success = await comprehensive_validation_framework.initialize()
    if not validation_init_success:
        print("❌ Failed to initialize validation framework")
        return False
    
    print("✅ Validation framework initialized successfully")
    
    # Run enhanced market analysis pipeline with orchestrator
    start_time = time.time()
    
    try:
        print("🎯 Starting enhanced market analysis pipeline with orchestrator...")
        
        # Use the step orchestrator for comprehensive execution
        result = await run_market_analysis_orchestrator(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if result.get('success', False):
            print("\n🎉 ENHANCED MARKET ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All enhanced market analysis steps completed:")
            print("   ✅ Data collection with validation")
            print("   ✅ HMM clustering with comprehensive validation")
            print("   ✅ Feature engineering with data protection")
            print("   ✅ Step orchestration with proper flow control")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Run comprehensive validation on the results
            print("🔍 Running comprehensive validation on pipeline results...")
            validation_reports = await validate_pipeline_comprehensive(
                result, ValidationLevel.COMPREHENSIVE
            )
            
            # Print validation summary
            validation_summary = comprehensive_validation_framework.get_validation_summary(validation_reports)
            print(f"📊 Validation Summary:")
            print(f"   Overall result: {validation_summary['overall_result']}")
            print(f"   Success rate: {validation_summary['success_rate']:.1f}%")
            print(f"   Total checks: {validation_summary['total_checks']}")
            print(f"   Passed: {validation_summary['total_passed']}")
            print(f"   Failed: {validation_summary['total_failed']}")
            print(f"   Warnings: {validation_summary['total_warnings']}")
            
            # Save comprehensive results
            results_file = Path(data_dir) / f"enhanced_market_analysis_results_{symbol}_{timeframe}.json"
            comprehensive_results = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'config': config,
                'execution_time': total_time,
                'success': True,
                'pipeline_result': result,
                'validation_reports': {
                    name: {
                        'overall_result': report.overall_result.value,
                        'total_checks': report.total_checks,
                        'passed_checks': report.passed_checks,
                        'failed_checks': report.failed_checks,
                        'warning_checks': report.warning_checks,
                        'execution_time': report.execution_time,
                        'summary': report.summary,
                        'recommendations': report.recommendations
                    }
                    for name, report in validation_reports.items()
                },
                'validation_summary': validation_summary,
                'timestamp': result.get('timestamp')
            }
            
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            print(f"💾 Comprehensive results saved to: {results_file}")
            
            # Print recommendations
            all_recommendations = []
            for report in validation_reports.values():
                all_recommendations.extend(report.recommendations)
            
            if all_recommendations:
                print("\n📋 Recommendations:")
                for i, rec in enumerate(set(all_recommendations), 1):
                    print(f"   {i}. {rec}")
            
        else:
            print("\n❌ ENHANCED MARKET ANALYSIS FAILED!")
            print("=" * 80)
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Still run validation on partial results if available
            if 'execution_results' in result:
                print("🔍 Running validation on partial results...")
                try:
                    validation_reports = await validate_pipeline_comprehensive(
                        result, ValidationLevel.BASIC
                    )
                    validation_summary = comprehensive_validation_framework.get_validation_summary(validation_reports)
                    print(f"📊 Partial Validation Summary:")
                    print(f"   Overall result: {validation_summary['overall_result']}")
                    print(f"   Success rate: {validation_summary['success_rate']:.1f}%")
                except Exception as validation_error:
                    print(f"⚠️ Validation of partial results failed: {validation_error}")
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 ENHANCED MARKET ANALYSIS FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        
        # Log the exception
        logger.exception(f"Enhanced market analysis failed: {e}")
        raise

if __name__ == "__main__":
    # Run the enhanced market analysis pipeline
    asyncio.run(main())