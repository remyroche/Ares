#!/usr/bin/env python3
"""
Test script for Enhanced Regime Data Splitting Step.

This script demonstrates the enhanced regime data splitting functionality with:
- BaseStep integration
- Multi-timeframe support (1h and 15m)
- Regime probability tagging
- Comprehensive regime metadata and statistics
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.market_analysis.enhanced_regime_data_splitting_step import EnhancedRegimeDataSplittingStep
from src.utils.tprint import tprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_enhanced_regime_data_splitting():
    """Test the enhanced regime data splitting step."""
    
    tprint("🧪 Starting Enhanced Regime Data Splitting Test", "INFO")
    
    # Initialize the enhanced step
    step = EnhancedRegimeDataSplittingStep()
    
    # Test configuration
    test_config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframes': ['1h', '15m'],
        'execution_mode': 'light'
    }
    
    tprint(f"📋 Test Configuration: {test_config}", "INFO")
    
    try:
        # Execute the enhanced regime data splitting step
        tprint("🚀 Executing Enhanced Regime Data Splitting Step...", "INFO")
        
        result = await step.execute(test_config)
        
        # Display results
        if result['success']:
            tprint("✅ Enhanced Regime Data Splitting completed successfully!", "SUCCESS")
            
            # Display execution summary
            execution_summary = result['metrics'].get('execution_summary', {})
            tprint(f"📊 Execution Summary:", "INFO")
            tprint(f"   - Total timeframes: {execution_summary.get('total_timeframes', 0)}", "INFO")
            tprint(f"   - Successful timeframes: {execution_summary.get('successful_timeframes', 0)}", "INFO")
            tprint(f"   - Failed timeframes: {execution_summary.get('failed_timeframes', 0)}", "INFO")
            
            # Display successful timeframes
            successful_timeframes = result.get('successful_timeframes', [])
            if successful_timeframes:
                tprint(f"✅ Successful timeframes: {successful_timeframes}", "SUCCESS")
            
            # Display failed timeframes
            failed_timeframes = result.get('failed_timeframes', [])
            if failed_timeframes:
                tprint(f"⚠️ Failed timeframes: {failed_timeframes}", "WARNING")
            
            # Display artifacts
            artifacts = result.get('artifacts', {})
            if artifacts:
                tprint(f"📁 Generated artifacts: {len(artifacts)} files", "INFO")
                for artifact_name, artifact_path in artifacts.items():
                    if artifact_name != 'cross_timeframe_analysis':
                        tprint(f"   - {artifact_name}: {artifact_path}", "INFO")
            
            # Display cross-timeframe analysis
            cross_timeframe_analysis = artifacts.get('cross_timeframe_analysis', {})
            if cross_timeframe_analysis:
                tprint("🔍 Cross-Timeframe Analysis:", "INFO")
                summary = cross_timeframe_analysis.get('cross_timeframe_summary', {})
                tprint(f"   - Total timeframes: {summary.get('total_timeframes', 0)}", "INFO")
                tprint(f"   - Successful timeframes: {summary.get('successful_timeframes', 0)}", "INFO")
                tprint(f"   - Failed timeframes: {summary.get('failed_timeframes', 0)}", "INFO")
                
                # Display timeframe comparison
                comparison = cross_timeframe_analysis.get('timeframe_comparison', {})
                if comparison:
                    regime_counts = comparison.get('regime_counts', {})
                    if regime_counts:
                        tprint("   - Regime counts by timeframe:", "INFO")
                        for tf, count in regime_counts.items():
                            tprint(f"     * {tf}: {count} regimes", "INFO")
                    
                    sample_counts = comparison.get('sample_counts', {})
                    if sample_counts:
                        tprint("   - Sample counts by timeframe:", "INFO")
                        for tf, count in sample_counts.items():
                            tprint(f"     * {tf}: {count} samples", "INFO")
                
                # Display regime consistency
                consistency = cross_timeframe_analysis.get('regime_consistency', {})
                if consistency:
                    tprint("   - Regime consistency metrics:", "INFO")
                    tprint(f"     * Mean regime count: {consistency.get('regime_count_mean', 0):.2f}", "INFO")
                    tprint(f"     * Std regime count: {consistency.get('regime_count_std', 0):.2f}", "INFO")
                    tprint(f"     * Coefficient of variation: {consistency.get('regime_count_cv', 0):.2f}", "INFO")
            
            # Display aggregate statistics
            aggregate_stats = result['metrics'].get('aggregate_statistics', {})
            if aggregate_stats:
                tprint("📈 Aggregate Statistics:", "INFO")
                tprint(f"   - Total samples across timeframes: {aggregate_stats.get('total_samples_across_timeframes', 0)}", "INFO")
                tprint(f"   - Average samples per timeframe: {aggregate_stats.get('average_samples_per_timeframe', 0):.2f}", "INFO")
                tprint(f"   - Total regimes across timeframes: {aggregate_stats.get('total_regimes_across_timeframes', 0)}", "INFO")
                tprint(f"   - Average regimes per timeframe: {aggregate_stats.get('average_regimes_per_timeframe', 0):.2f}", "INFO")
            
            # Display timeframe-specific results
            timeframe_results = result.get('timeframe_results', {})
            if timeframe_results:
                tprint("⏰ Timeframe-Specific Results:", "INFO")
                for tf, tf_result in timeframe_results.items():
                    if tf_result.get('success', False):
                        metrics = tf_result.get('metrics', {})
                        tprint(f"   - {tf}:", "INFO")
                        tprint(f"     * Total samples: {metrics.get('total_samples', 0)}", "INFO")
                        tprint(f"     * Regime count: {metrics.get('regime_count', 0)}", "INFO")
                        
                        # Display data quality metrics
                        data_quality = metrics.get('data_quality', {})
                        if data_quality:
                            tprint(f"     * Data completeness: {data_quality.get('completeness', 0):.2%}", "INFO")
                            tprint(f"     * Duplicate rows: {data_quality.get('duplicate_rows', 0)}", "INFO")
                            tprint(f"     * Memory usage: {data_quality.get('memory_usage_mb', 0):.2f} MB", "INFO")
                        
                        # Display regime metrics
                        regime_metrics = metrics.get('regime_metrics', {})
                        if regime_metrics:
                            tprint(f"     * Regime distribution:", "INFO")
                            for regime_id, regime_info in regime_metrics.items():
                                total_samples = regime_info.get('total_samples', 0)
                                train_samples = regime_info.get('train_samples', 0)
                                val_samples = regime_info.get('validation_samples', 0)
                                test_samples = regime_info.get('test_samples', 0)
                                tprint(f"       - Regime {regime_id}: {total_samples} total ({train_samples} train, {val_samples} val, {test_samples} test)", "INFO")
                        
                        # Display probability metrics
                        prob_metrics = metrics.get('probability_metrics', {})
                        if prob_metrics:
                            tprint(f"     * Probability metrics:", "INFO")
                            tprint(f"       - Mean probability: {prob_metrics.get('mean_probability', 0):.4f}", "INFO")
                            tprint(f"       - Std probability: {prob_metrics.get('std_probability', 0):.4f}", "INFO")
                            tprint(f"       - Mean confidence: {prob_metrics.get('confidence_mean', 0):.4f}", "INFO")
                            tprint(f"       - Mean uncertainty: {prob_metrics.get('uncertainty_mean', 0):.4f}", "INFO")
                    else:
                        tprint(f"   - {tf}: FAILED - {tf_result.get('error', 'Unknown error')}", "ERROR")
            
            # Save detailed results to file
            results_file = Path("enhanced_regime_data_splitting_test_results.json")
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            tprint(f"💾 Detailed results saved to: {results_file}", "INFO")
            
        else:
            tprint(f"❌ Enhanced Regime Data Splitting failed: {result.get('error', 'Unknown error')}", "ERROR")
            return False
            
    except Exception as e:
        tprint(f"❌ Test failed with exception: {str(e)}", "ERROR")
        logger.exception("Test failed")
        return False
    
    tprint("🎉 Enhanced Regime Data Splitting Test completed!", "SUCCESS")
    return True


async def test_base_step_integration():
    """Test BaseStep integration features."""
    
    tprint("🔧 Testing BaseStep Integration", "INFO")
    
    # Initialize the enhanced step
    step = EnhancedRegimeDataSplittingStep()
    
    # Test BaseStep properties
    tprint(f"📋 Step name: {step.step_name}", "INFO")
    tprint(f"📋 Logger: {step.logger.name}", "INFO")
    tprint(f"📋 Artifact manager: {type(step.artifact_manager).__name__}", "INFO")
    
    # Test supported timeframes
    tprint(f"📋 Supported timeframes: {step.supported_timeframes}", "INFO")
    
    # Test regime configuration
    tprint(f"📋 Regime config: {step.regime_config}", "INFO")
    
    tprint("✅ BaseStep integration test completed", "SUCCESS")
    return True


async def test_multi_timeframe_support():
    """Test multi-timeframe support functionality."""
    
    tprint("⏰ Testing Multi-Timeframe Support", "INFO")
    
    # Initialize the enhanced step
    step = EnhancedRegimeDataSplittingStep()
    
    # Test with different timeframe combinations
    test_configs = [
        {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframes': ['1h'],
            'execution_mode': 'light'
        },
        {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframes': ['15m'],
            'execution_mode': 'light'
        },
        {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframes': ['1h', '15m'],
            'execution_mode': 'light'
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        tprint(f"🔄 Test {i}: {config['timeframes']} timeframes", "INFO")
        
        try:
            result = await step.execute(config)
            
            if result['success']:
                successful_tfs = result.get('successful_timeframes', [])
                failed_tfs = result.get('failed_timeframes', [])
                tprint(f"   ✅ Success: {successful_tfs}, Failed: {failed_tfs}", "SUCCESS")
            else:
                tprint(f"   ❌ Failed: {result.get('error', 'Unknown error')}", "ERROR")
                
        except Exception as e:
            tprint(f"   ❌ Exception: {str(e)}", "ERROR")
    
    tprint("✅ Multi-timeframe support test completed", "SUCCESS")
    return True


async def main():
    """Main test function."""
    
    tprint("🚀 Starting Enhanced Regime Data Splitting Tests", "INFO")
    tprint("=" * 60, "INFO")
    
    # Test BaseStep integration
    await test_base_step_integration()
    tprint("", "INFO")
    
    # Test multi-timeframe support
    await test_multi_timeframe_support()
    tprint("", "INFO")
    
    # Test full functionality
    await test_enhanced_regime_data_splitting()
    
    tprint("=" * 60, "INFO")
    tprint("🎉 All tests completed!", "SUCCESS")


if __name__ == "__main__":
    asyncio.run(main())
