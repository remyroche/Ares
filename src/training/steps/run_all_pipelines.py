#!/usr/bin/env python3
"""Main Orchestrator for All Training Pipelines.

This module provides the main interface to run all training pipelines:
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
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all pipeline modules
from src.training.steps.data_collection import run_data_collection_pipeline
from src.training.steps.market_analysis import run_market_analysis_pipeline
from src.training.steps.model_training import run_model_training_pipeline
from src.training.steps.optimisation import run_optimisation_pipeline
from src.training.steps.backtesting import run_backtesting_pipeline

async def run_all_pipelines(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **config: Dict[str, Any]
) -> bool:
    """Run all training pipelines in sequence."""
    
    print("🚀 COMPLETE TRADING PIPELINE EXECUTION")
    print("=" * 100)
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print("=" * 100)
    
    total_start_time = time.time()
    results = {}
    
    # Pipeline configurations
    pipeline_configs = {
        'data_collection': {
            'force_rerun': config.get('force_rerun', True),
            'quality_checks': config.get('quality_checks', True),
            'validate_data': config.get('validate_data', True),
            'convert_format': config.get('convert_format', True),
        },
        'market_analysis': {
            'force_rerun': config.get('force_rerun', True),
            'hmm_clustering': config.get('hmm_clustering', True),
            'regime_splitting': config.get('regime_splitting', True),
            'feature_engineering': config.get('feature_engineering', True),
            'matrix_operations': config.get('matrix_operations', True),
            'feature_selection': config.get('feature_selection', True),
        },
        'model_training': {
            'force_rerun': config.get('force_rerun', True),
            'hmm_training': config.get('hmm_training', True),
            'regime_intelligence': config.get('regime_intelligence', True),
            'analyst_creation': config.get('analyst_creation', True),
            'analyst_enhancement': config.get('analyst_enhancement', True),
            'ensemble_creation': config.get('ensemble_creation', True),
            'tactician_training': config.get('tactician_training', True),
        },
        'optimisation': {
            'force_rerun': config.get('force_rerun', True),
            'confidence_calibration': config.get('confidence_calibration', True),
            'parameter_optimization': config.get('parameter_optimization', True),
        },
        'backtesting': {
            'force_rerun': config.get('force_rerun', True),
            'walk_forward_validation': config.get('walk_forward_validation', True),
            'monte_carlo_validation': config.get('monte_carlo_validation', True),
            'ab_testing': config.get('ab_testing', True),
            'model_saving': config.get('model_saving', True),
        }
    }
    
    # Pipeline execution order
    pipelines = [
        ('Data Collection', run_data_collection_pipeline, pipeline_configs['data_collection']),
        ('Market Analysis', run_market_analysis_pipeline, pipeline_configs['market_analysis']),
        ('Model Training', run_model_training_pipeline, pipeline_configs['model_training']),
        ('Optimization', run_optimisation_pipeline, pipeline_configs['optimisation']),
        ('Backtesting', run_backtesting_pipeline, pipeline_configs['backtesting']),
    ]
    
    # Execute pipelines in sequence
    for pipeline_name, pipeline_func, pipeline_config in pipelines:
        print(f"\n🔄 Starting {pipeline_name} Pipeline...")
        print("-" * 80)
        
        pipeline_start_time = time.time()
        
        try:
            success = await pipeline_func(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                **pipeline_config
            )
            
            pipeline_time = time.time() - pipeline_start_time
            results[pipeline_name] = {
                'success': success,
                'execution_time': pipeline_time,
                'error': None
            }
            
            if success:
                print(f"✅ {pipeline_name} Pipeline completed successfully in {pipeline_time:.2f} seconds")
            else:
                print(f"❌ {pipeline_name} Pipeline failed after {pipeline_time:.2f} seconds")
                
        except Exception as e:
            pipeline_time = time.time() - pipeline_start_time
            results[pipeline_name] = {
                'success': False,
                'execution_time': pipeline_time,
                'error': str(e)
            }
            print(f"💥 {pipeline_name} Pipeline failed with exception: {e}")
            print(f"⏱️ Execution time: {pipeline_time:.2f} seconds")
    
    # Final results
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 100)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 100)
    
    successful_pipelines = 0
    failed_pipelines = 0
    
    for pipeline_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{pipeline_name:20} | {status:10} | {result['execution_time']:8.2f}s")
        if result['error']:
            print(f"{'':20} | Error: {result['error']}")
        
        if result['success']:
            successful_pipelines += 1
        else:
            failed_pipelines += 1
    
    print("-" * 100)
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Successful Pipelines: {successful_pipelines}/{len(pipelines)}")
    print(f"Failed Pipelines: {failed_pipelines}/{len(pipelines)}")
    
    if failed_pipelines == 0:
        print("🎉 ALL PIPELINES COMPLETED SUCCESSFULLY!")
    else:
        print(f"⚠️  {failed_pipelines} PIPELINE(S) FAILED")
    
    print("=" * 100)
    
    # Save results
    results_file = Path(data_dir) / f"pipeline_results_{symbol}_{timeframe}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'total_execution_time': total_time,
            'successful_pipelines': successful_pipelines,
            'failed_pipelines': failed_pipelines,
            'results': results,
            'config': config
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    return failed_pipelines == 0

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