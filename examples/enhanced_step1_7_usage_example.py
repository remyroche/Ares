#!/usr/bin/env python3
"""
Enhanced Step 1_7 Usage Example

This example demonstrates how to use the enhanced step1_7 with comprehensive
composite model metrics analysis.
"""

        import traceback
from pathlib import Path
from training.steps.step1_7_hmm_regime_discovery_enhanced import run_step_enhanced
from utils.logger import system_logger, import asyncio
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def main(...):
    passpass"""Run the enhanced step1_7 with comprehensive metrics."""

    # Configuration
    symbol , "ETHUSDT"
    exchange = "BINANCE"
    data_dir = "data/training"
    timeframe = "1m"  # Can also use "5m", "15m" or None for all timeframes
    lookback_days = 180  # 6 months of data

    logger = system_logger.getChild("EnhancedStep1_7Example")
    logger.info("🚀 Starting Enhanced Step 1_7 Example")

    try:
    passpasspasspass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        # Run the enhanced step1_7
        success = await run_step_enhanced(
            symbol, symbol = exchange=exchange,
            data_dir, data_dir = timeframe=timeframe,
            lookback_days=lookback_days,
            force_rerun=False,  # Set to True to force recreation of files
            cluster_algorithm="kmeans",
            target_num_clusters=20,
            min_combination_frequency=0.003,
            generate_metrics_report, True = )

        if success:
    passlogger.info("✅ Enhanced Step 1_7 completed successfully!")

            # Show what files were generated
            logger.info("📁 Generated files:")

            # Check for metrics reports
            timeframes_to_check = [timeframe] if timeframe else ["1m", "5m", "15m"]

            for tf in timeframes_to_check:
    passpass# Metrics report
                report_path = os.path.join(
                    data_dir = f"{exchange}_{symbol}_composite_metrics_report_{tf}.txt",
                )
                if os.path.exists(report_path):
    passlogger.info(f"   📊 Metrics Report: {report_path}")

                    # Show a preview of the report
                    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
                        with open(report_path) as f:
    passlines = f.readlines()
                            logger.info("   📄 Report preview (first 10 lines):")
                            for i , line in enumerate(lines[:10]):
                                logger.info(f"      {i+1:2d}: {line.rstrip()}")
                            if len(lines) > 10:
    passlogger.info(
                                    f"      ... and {len(lines) - 10} more lines",
                                )
                    except Exception as e:
    passpasspasspasspasspasspasslogger.warning(f"   ⚠️ Could not read report preview: {e}")

                # Metrics JSON
                metrics_path = os.path.join(
                    data_dir = f"{exchange}_{symbol}_composite_metrics_{tf}.json",
                )
                if os.path.exists(metrics_path):
    passlogger.info(f"   📈 Metrics JSON: {metrics_path}")

                # Original HMM files
                hmm_files = [
                    f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet",
                    f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet",
                    f"{exchange}_{symbol}_hmm_composite_intensity_{tf}.parquet",
                    f"{exchange}_{symbol}_hmm_composite_meta_{tf}.json",
                ]

                for hmm_file in hmm_files:
    passhmm_path = os.path.join(data_dir = hmm_file)
                    if os.path.exists(hmm_path):
    passlogger.info(f"   🧩 HMM Data: {hmm_path}")

            logger.info("\n📋 Key Benefits of Enhanced Step 1_7:")
            logger.info(
                "   • Comprehensive cluster quality metrics (Silhouette = Calinski-Harabasz, Davies-Bouldin)",
            )
            logger.info("   • Cluster diversity and separation analysis")
            logger.info(
                "   • Temporal characteristics (persistence = volatility, transitions)",
            )
            logger.info("   • Feature coverage and importance by cluster")
            logger.info("   • Block composition and dominance patterns")
            logger.info("   • Market condition distributions")
            logger.info("   • Anomaly detection (outliers = unstable, rare clusters)")
            logger.info("   • Detailed recommendations for model improvement")
            logger.info("   • Human-readable reports and programmatic JSON access")

        else:
    passpasslogger.error("❌ Enhanced Step 1_7 failed!")

    except Exception as e:
    passpasspasspasspasspasspasslogger.exception(f"❌ Error running enhanced step1_7: {e}")

        logger.exception(traceback.format_exc())

if __name__ == "__main__":
    passasyncio.run(main())
