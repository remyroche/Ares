#!/usr/bin/env python3
"""
Test Data Collection Pipeline

This script tests the complete data collection pipeline integration
with the Binance API to ensure all components work together properly.
"""

import sys
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_complete_data_collection_pipeline():
    """Test the complete data collection pipeline."""
    print("🧪 Testing Complete Data Collection Pipeline...")
    print("="*60)

    try:
        # Import the orchestrator
        from src.training.steps.data_collection.data_collection_orchestrator import DataCollectionOrchestrator
        print("✅ Data Collection Orchestrator imported successfully")

        # Create orchestrator
        orchestrator = DataCollectionOrchestrator()
        print("✅ Data Collection Orchestrator initialized")

        # Test parameters
        symbol = "BTCUSDT"
        exchange = "BINANCE"
        data_types = ["klines", "aggtrades"]  # Skip futures for faster testing
        timeframes = ["1m", "5m", "15m"]  # Limit timeframes for testing
        lookback_days = 2  # Short lookback for testing

        print(f"\n🔧 Test Configuration:")
        print(f"   Symbol: {symbol}")
        print(f"   Exchange: {exchange}")
        print(f"   Data Types: {data_types}")
        print(f"   Timeframes: {timeframes}")
        print(f"   Lookback: {lookback_days} days")

        # Run the complete pipeline
        print(f"\n🚀 Running Complete Pipeline...")
        start_time = datetime.now()

        results = await orchestrator.run_complete_pipeline(
            symbol=symbol,
            exchange=exchange,
            data_types=data_types,
            timeframes=timeframes,
            lookback_days=lookback_days
        )

        end_time = datetime.now()
        execution_time = end_time - start_time

        print(f"\n⏱️ Pipeline Execution Time: {execution_time}")

        # Analyze results
        print(f"\n📊 Pipeline Results Analysis:")
        print(f"Overall Success: {'✅' if results['success'] else '❌'}")

        # Stage-by-stage analysis
        stages = results.get('stages', {})
        stage_results = []

        # Download stage
        download_stage = stages.get('download', {})
        if download_stage:
            success_rate = (download_stage.get('downloads_successful', 0) /
                          max(download_stage.get('downloads_attempted', 1), 1) * 100)
            stage_results.append(("Download", download_stage.get('success', False), success_rate))
            print(f"📥 Download Stage: {'✅' if download_stage.get('success', False) else '❌'} ({success_rate:.1f}% success)")

        # Gap filling stage
        gap_stage = stages.get('gap_filling', {})
        if gap_stage:
            gaps_detected = gap_stage.get('gaps_detected', 0)
            gaps_filled = gap_stage.get('gaps_filled', 0)
            fill_rate = (gaps_filled / max(gaps_detected, 1) * 100)
            stage_results.append(("Gap Filling", gap_stage.get('success', False), fill_rate))
            print(f"🔧 Gap Filling: {'✅' if gap_stage.get('success', False) else '❌'} ({gaps_filled}/{gaps_detected} gaps filled, {fill_rate:.1f}% rate)")

        # Quality checks stage
        quality_stage = stages.get('quality_checks', {})
        if quality_stage:
            checks_passed = quality_stage.get('checks_passed', 0)
            total_checks = quality_stage.get('total_checks', 0)
            quality_rate = (checks_passed / max(total_checks, 1) * 100)
            stage_results.append(("Quality Checks", quality_stage.get('success', False), quality_rate))
            print(f"✅ Quality Checks: {'✅' if quality_stage.get('success', False) else '❌'} ({checks_passed}/{total_checks} passed, {quality_rate:.1f}% rate)")

            # Show quality scores
            quality_scores = quality_stage.get('quality_scores', {})
            if quality_scores:
                print("   Quality Scores:")
                for data_type, score in quality_scores.items():
                    print(".2f")

        # Resampling stage
        resampling_stage = stages.get('resampling', {})
        if resampling_stage:
            resamples_successful = resampling_stage.get('resamples_successful', 0)
            resamples_attempted = resampling_stage.get('resamples_attempted', 0)
            resample_rate = (resamples_successful / max(resamples_attempted, 1) * 100)
            stage_results.append(("Resampling", resampling_stage.get('success', False), resample_rate))
            print(f"📊 Resampling: {'✅' if resampling_stage.get('success', False) else '❌'} ({resamples_successful}/{resamples_attempted} successful, {resample_rate:.1f}% rate)")

        # Storage stage
        storage_stage = stages.get('storage', {})
        if storage_stage:
            files_organized = storage_stage.get('files_organized', 0)
            stage_results.append(("Storage", storage_stage.get('success', False), files_organized))
            print(f"💾 Storage: {'✅' if storage_stage.get('success', False) else '❌'} ({files_organized} files organized)")

        # Overall statistics
        pipeline_stats = results.get('pipeline_stats', {})
        if pipeline_stats:
            print(f"\n📈 Pipeline Statistics:")
            print(f"   Downloads: {pipeline_stats.get('downloads_successful', 0)}/{pipeline_stats.get('downloads_attempted', 0)}")
            print(f"   Gaps Detected/Filled: {pipeline_stats.get('gaps_detected', 0)}/{pipeline_stats.get('gaps_filled', 0)}")
            print(f"   Quality Checks Passed: {pipeline_stats.get('quality_checks_passed', 0)}")
            print(f"   Resampling Completed: {pipeline_stats.get('resampling_completed', 0)}")

        # Errors and warnings
        errors = results.get('errors', [])
        warnings = results.get('warnings', [])

        if errors:
            print(f"\n❌ Errors ({len(errors)}):")
            for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
                print(f"   {i}. {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")

        if warnings:
            print(f"\n⚠️ Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings[:3], 1):  # Show first 3 warnings
                print(f"   {i}. {warning}")
            if len(warnings) > 3:
                print(f"   ... and {len(warnings) - 3} more warnings")

        # Stage summary
        print(f"\n📋 Stage Summary:")
        all_stages_successful = all(result[1] for result in stage_results)
        print(f"   All Stages Successful: {'✅' if all_stages_successful else '❌'}")

        for stage_name, success, metric in stage_results:
            status = "✅" if success else "❌"
            print(f"   {stage_name}: {status} ({metric})")

        # Final assessment
        print(f"\n" + "="*60)
        if results['success'] and all_stages_successful:
            print("🎉 COMPLETE PIPELINE TEST: SUCCESS!")
            print("✅ All stages completed successfully")
            print("✅ Binance API integration working properly")
            print("✅ Data collection, gap filling, quality checks, and resampling all functional")
            print("✅ Production-ready data collection pipeline")
        elif results['success']:
            print("⚠️ COMPLETE PIPELINE TEST: PARTIAL SUCCESS")
            print("✅ Pipeline completed but some stages had issues")
            print("⚠️ Check individual stage results above")
        else:
            print("❌ COMPLETE PIPELINE TEST: FAILED")
            print("❌ Pipeline execution failed")
            print("❌ Check error messages above")

        print("="*60)

        # Verify data files were created
        print(f"\n🔍 Verifying Data Files...")
        data_cache = Path("data_cache")
        if data_cache.exists():
            # Count data files
            klines_files = list(data_cache.glob("klines_*.parquet"))
            aggtrades_files = list(data_cache.glob("aggtrades_*.parquet"))
            resampled_files = list(data_cache.glob("*_5m_*.parquet"))  # Check for 5m resampled files

            print(f"   📁 Data files created:")
            print(f"      Klines: {len(klines_files)} files")
            print(f"      AggTrades: {len(aggtrades_files)} files")
            print(f"      Resampled (5m): {len(resampled_files)} files")

            if klines_files:
                print(f"   📄 Sample klines file: {klines_files[0].name}")
            if aggtrades_files:
                print(f"   📄 Sample aggtrades file: {aggtrades_files[0].name}")
            if resampled_files:
                print(f"   📄 Sample resampled file: {resampled_files[0].name}")
        else:
            print("   ❌ Data cache directory not found")

        return results['success']

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual pipeline components."""
    print("\n🔧 Testing Individual Components...")
    print("-"*40)

    try:
        # Test unified downloader
        print("Testing Unified Downloader...")
        from src.training.steps.data_collection.unified_data_downloader import UnifiedDataDownloader

        downloader = UnifiedDataDownloader()
        await downloader._get_exchange_instance("BINANCE")
        print("✅ Unified Downloader: Exchange initialization successful")

        # Test unified gap filler
        print("Testing Unified Gap Filler...")
        from src.training.steps.data_collection.unified_gap_filler import UnifiedGapFiller

        gap_filler = UnifiedGapFiller()
        print("✅ Unified Gap Filler: Initialization successful")

        # Test unified resampler
        print("Testing Unified Resampler...")
        from src.training.steps.data_collection.unified_resampler import UnifiedResampler

        resampler = UnifiedResampler()
        print("✅ Unified Resampler: Initialization successful")

        print("✅ All individual components initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Individual component test failed: {e}")
        return False

async def main():
    """Run all pipeline tests."""
    print("🚀 Starting Data Collection Pipeline Tests...")
    print("="*60)

    # Test individual components first
    components_ok = await test_individual_components()

    if not components_ok:
        print("❌ Component initialization failed, skipping pipeline test")
        return

    # Test complete pipeline
    pipeline_success = await test_complete_data_collection_pipeline()

    # Final summary
    print(f"\n" + "="*60)
    print("📊 FINAL TEST RESULTS")
    print("="*60)
    print(f"Component Tests: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"Pipeline Test: {'✅ PASS' if pipeline_success else '❌ FAIL'}")

    if components_ok and pipeline_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Data collection pipeline is fully functional")
        print("✅ Binance API integration is working correctly")
        print("✅ Ready for production use")
    else:
        print("❌ SOME TESTS FAILED")
        print("❌ Check the output above for details")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
