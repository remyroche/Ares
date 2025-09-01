#!/usr / bin / env python3
"""Step 1 Runner Script.

This script demonstrates how to use the step1 module to:
    pass
1. Detect missing data gaps
2. Validate and fix aggtrades format
3. Resample data to multiple timeframes
4. Ensure step01_5 compatibility
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from src.training.steps.step1 import (
AggtradesValidator,
DataGapDetector,
DataPreparation,
MissingDataDownloaderAndGapFiller,
Step1Orchestrator,
)
from src.utils.logger import system_logger

# Add project root to path
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger, system_logger.getChild("Step1Runner")

def main() -> None:
    """Main function to run step1 processes."""
start_time, datetime.now()

logger.info("🚀 STEP1 LAUNCHER STARTING")
logger.info("=" * 80)

parser, argparse.ArgumentParser(description="Step 1 Data Collection and Validation")
parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
parser.add_argument("--exchange", default="BINANCE", help="Exchange name")
parser.add_argument("--start - date", help="Start date (YYYY - MM - DD)")
parser.add_argument("--end - date", help="End date (YYYY - MM - DD)")
parser.add_argument("--no - auto - fix", action="store_true", help="Disable auto - fixing")
parser.add_argument("--mode", choices=["complete", "gap - detection", "validation", "preparation", "health - check", "status", "download - missing"],
default="complete", help="Operation mode")

args, parser.parse_args()

logger.info(f"🎯 TARGET: {args.exchange}_{args.symbol}")
logger.info(f"📅 Date range: {args.start_date} to {args.end_date}")
logger.info(f"🔧 Auto - fix: {'Disabled' if args.no_auto_fix else 'Enabled'}")
logger.info(f"⚙️  Mode: {args.mode}")
logger.info("-" * 60)

# Parse dates
start_date, None
end_date, None
if args.start_date:
        start_date, datetime.strptime(args.start_date, "%Y-%m-%d")
if args.end_date:
        end_date, datetime.strptime(args.end_date, "%Y-%m-%d")

# Initialize orchestrator
orchestrator, Step1Orchestrator()

if args.mode == "complete":
        # Run complete step1 process
results, asyncio.run(orchestrator.run_complete_step1(
symbol = args.symbol,
exchange = args.exchange,
start_date = start_date,
end_date = end_date,
auto_fix = not args.no_auto_fix
))

# Print report
if "report" in results:
            print(results["report"])

# Print summary
end_time, datetime.now()
execution_time, end_time - start_time

logger.info("=" * 80)
logger.info("📊 STEP1 LAUNCHER SUMMARY")
logger.info(f"⏱️  Total execution time: {execution_time}")
logger.info(f"🎯 Target: {args.exchange}_{args.symbol}")
logger.info(f"⚙️  Mode: {args.mode}")

if results["success"]:
            logger.info("✅ STEP1 COMPLETED SUCCESSFULLY!")
print("✅ Step1 completed successfully!")
else:
            logger.error("❌ STEP1 COMPLETED WITH ERRORS!")
print("❌ Step1 completed with errors:")
for error in results["errors"]:
                logger.error(f"  - {error}")
print(f"  - {error}")

logger.info("=" * 80)

elif args.mode == "gap - detection":
        # Run gap detection only
gap_detector, DataGapDetector()

# Detect missing data
missing_data, gap_detector.detect_missing_data(args.symbol, args.exchange, start_date, end_date)

# Detect aggtrades gaps
aggtrades_gaps, gap_detector.detect_aggtrades_gaps(args.symbol, args.exchange)

# Generate report
gap_detector.generate_missing_data_report(args.symbol, args.exchange)

# Print gap details
if aggtrades_gaps:
            print(f"Found {len(aggtrades_gaps)} gaps in aggtrades data:")
for gap in aggtrades_gaps[:10]:  # Show first 10
print(f"  - {gap['file']}: {gap['gap_start']} to {gap['gap_end']}")
if len(aggtrades_gaps) > 10:
                print(f"  ... and {len(aggtrades_gaps) - 10} more gaps")

elif args.mode == "validation":
        # Run validation only
validator, AggtradesValidator()

# Validate all aggtrades
validation_results, validator.validate_all_aggtrades(
args.symbol, args.exchange, auto_fix = not args.no_auto_fix
)

# Generate report
validator.generate_validation_report(args.symbol, args.exchange)

# Print summary
print(f"Validation completed: {validation_results['valid_files']} valid, {validation_results['invalid_files']} invalid")

elif args.mode == "preparation":
        # Run preparation only
preparation, DataPreparation()

# Prepare data for step01_5
preparation_results, preparation.prepare_for_step01_5(args.symbol, args.exchange)

if preparation_results["ready"]:
            print("✅ Data preparation completed successfully")
else:
            print("❌ Data preparation encountered issues:")
for issue in preparation_results["issues"]:
                print(f"  - {issue}")

elif args.mode == "health - check":
        # Run health check only
health_result, orchestrator.quick_health_check(args.symbol, args.exchange)

if health_result["healthy"]:
            print("✅ Health check passed")
else:
            print("❌ Health check found issues:")
for issue in health_result["issues"]:
                print(f"  - {issue}")

for recommendation in health_result["recommendations"]:
            print(f"  💡 {recommendation}")

elif args.mode == "status":
        # Show current status
status, orchestrator.get_step1_status(args.symbol, args.exchange)

print(f"Status: {status['overall_status']}")
print(f"Aggtrades files: {status['data_available']['aggtrades']}")
print(f"Klines files: {status['data_available']['klines']}")
print("Resampled data:")
for timeframe, available in status["resampled_data"].items():
            print(f"  - {timeframe}: {'✅' if available else '❌'}")

elif args.mode == "download - missing":
        # Run missing data download only
downloader, MissingDataDownloaderAndGapFiller()

# Run async download process
download_results, asyncio.run(
downloader.download_all_missing_data(args.symbol, args.exchange, end_date),
)

# Print report
if "report" in download_results:
            print(download_results["report"])

# Print summary
if download_results["success"]:
            print("✅ Download completed successfully!")
else:
            print("❌ Download completed with errors:")
for error in download_results["errors"]:
                print(f"  - {error}")

if __name__ == "__main__":
    main()