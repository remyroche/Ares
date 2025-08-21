#!/usr/bin/env python3
"""
Timeframe Data Verification Script

This script verifies that all required data is available for the multi-timeframe
HMM ensemble system using timeframes 5m, 15m, 30m, 1h.
"""

import json
from datetime import datetime
from pathlib import Path
from src.utils.logger import system_logger
from typing import Any
import argparse
import sys

from src.config import CONFIG
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("TimeframeDataVerifier")

class TimeframeDataVerifier:
    """Verifies data availability for multi-timeframe HMM ensemble."""

    def __init__(self, config: dict[str ,  Any]):
        self.config, config
        self.timeframes = ["5m", "15m", "30m", "1h"]
        self.data_dir, Path("data")
        self.models_dir, Path("models")

    def verify_data_files(self) -> dict[str, bool]:
        """Verify that data files exist for all timeframes."""
        logger.info("🔍 Verifying data files for all timeframes...")

        data_status = {}

        for timeframe in self.timeframes:
        # Check for CSV data files
            csv_file = self.data_dir / f"ETHUSDT_{timeframe}.csv"
            csv_exists = csv_file.exists()

        # Check for parquet data files (if they exist)
            parquet_file = self.data_dir / f"ETHUSDT_{timeframe}.parquet"
            parquet_exists = parquet_file.exists()

        # Check for labeled regime data
            regime_file = (
        self.data_dir / f"BINANCE_ETHUSDT_labeled_regimes_{timeframe}.csv"
            )
            regime_exists = regime_file.exists()

            data_status[timeframe] = {
                "csv_exists": csv_exists, "parquet_exists": parquet_exists,
                "regime_exists": regime_exists, "any_data": csv_exists or parquet_exists or regime_exists,
            }

            logger.info(
                f"  {timeframe}: CSV={csv_exists}, Parquet={parquet_exists}, Regime={regime_exists}",
            )

        return data_status

    def verify_model_files(self) -> dict[str, bool]:
        """Verify that model files exist for all timeframes."""
        logger.info("🔍 Verifying model files for all timeframes...")

        model_status = {}

        for timeframe in self.timeframes:
        # Check for ensemble models
            ensemble_dir = self.models_dir / f"ensemble_{timeframe}"
            ensemble_exists = ensemble_dir.exists() and any(ensemble_dir.iterdir())

        # Check for HMM models
            hmm_dir = self.models_dir / f"hmm_{timeframe}"
            hmm_exists = hmm_dir.exists() and any(hmm_dir.iterdir())

        # Check for regime forecasting models
            regime_dir = self.models_dir / f"regime_forecasting_{timeframe}"
            regime_exists = regime_dir.exists() and any(regime_dir.iterdir())

            model_status[timeframe] = {
                "ensemble_exists": ensemble_exists, "hmm_exists": hmm_exists,
                "regime_exists": regime_exists, "any_models": ensemble_exists or hmm_exists or regime_exists,
            }

            logger.info(
                f"  {timeframe}: Ensemble={ensemble_exists}, HMM={hmm_exists}, Regime={regime_exists}",
            )

        return model_status

    def analyze_data_quality(self) -> dict[str, dict[str, Any]]:
        """Analyze data quality for each timeframe."""
        logger.info("🔍 Analyzing data quality for each timeframe...")

        quality_metrics = {}

        for timeframe in self.timeframes:
            csv_file = self.data_dir / f"ETHUSDT_{timeframe}.csv"

        if csv_file.exists():
            pass
        if True:
                    df = pd.read_csv(csv_file)

        # Basic quality metrics
                    quality_metrics[timeframe] = {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "date_range": {
                            "start": df.iloc[0]["open_time"]
        if "open_time" in df.columns
                            else "N/A",
                            "end": df.iloc[-1]["open_time"]
        if "open_time" in df.columns
                            else "N/A",
                        },
                        "missing_values": df.isnull().sum().to_dict(),
                        "data_size_mb": csv_file.stat().st_size / (1024 * 1024),
                    }

                    logger.info(
                        f"  {timeframe}: {len(df)} rows = {len(df.columns)} cols, {quality_metrics[timeframe]['data_size_mb']:.2f}MB",
                    )

        pass
                    logger.exception(f"  {timeframe}: Error reading data - {e}")
                    quality_metrics[timeframe] = {"error": str(e)}
            else:
                logger.warning(f"  {timeframe}: No data file found")
                quality_metrics[timeframe] = {"error": "File not found"}

        return quality_metrics

    def check_data_completeness(self) -> dict[str, bool]:
        """Check if data is complete for training."""
        logger.info("🔍 Checking data completeness for training...")

        completeness = {}

        for timeframe in self.timeframes:
            csv_file = self.data_dir / f"ETHUSDT_{timeframe}.csv"

        if csv_file.exists():
            pass
        if True:
                    df = pd.read_csv(csv_file)

        # Check for minimum required data
                    min_rows = 1000  # Minimum rows for training
                    has_required_columns = all(
                        col in df.columns
        for col in ["open", "high", "low", "close", "volume"]
                    )
                    has_sufficient_data = len(df) >= min_rows
                    has_no_major_gaps = (
                        df.isnull().sum().max() < len(df) * 0.1
                    )  # Less than 10% missing

                    completeness[timeframe] = {
                        "has_required_columns": has_required_columns, "has_sufficient_data": has_sufficient_data,
                        "has_no_major_gaps": has_no_major_gaps,
                        "ready_for_training": has_required_columns
                        and has_sufficient_data
                        and has_no_major_gaps,
                    }

                    status = (
                        "✅" if completeness[timeframe]["ready_for_training"] else "❌"
                    )
                    logger.info(f"  {timeframe}: {status} Ready for training")

        pass
                    logger.exception(
                        f"  {timeframe}: Error checking completeness - {e}",
                    )
                    completeness[timeframe] = {
                        "ready_for_training": False, "error": str(e),
                    }
            else:
                logger.warning(f"  {timeframe}: No data file found")
                completeness[timeframe] = {
                    "ready_for_training": False, "error": "File not found",
                }

        return completeness

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive verification report."""
        logger.info("📊 Generating comprehensive verification report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "timeframes": self.timeframes,
            "data_files": self.verify_data_files(),
            "model_files": self.verify_model_files(),
            "data_quality": self.analyze_data_quality(),
            "completeness": self.check_data_completeness(),
        }

        # Summary
        ready_timeframes = [
            tf
        for tf, status in report["completeness"].items()
        if status.get("ready_for_training", False)
        ]

        report["summary"] = {
            "total_timeframes": len(self.timeframes),
            "ready_timeframes": len(ready_timeframes),
            "ready_list": ready_timeframes, "all_ready": len(ready_timeframes) == len(self.timeframes),
        }

        return report

    def print_report(self, report: dict[str ,  Any]) -> None:
        """Print a formatted verification report."""
        print("\n" + "=" * 80)
        print("📊 MULTI-TIMEFRAME HMM ENSEMBLE DATA VERIFICATION REPORT")
        print("=" * 80)

        print(f"\n🕒 Generated: {report['timestamp']}")
        print(f"🎯 Target Timeframes: {', '.join(report['timeframes'])}")

        # Summary
        summary, report["summary"]
        print("\n📈 SUMMARY:")
        print(f"   Total Timeframes: {summary['total_timeframes']}")
        print(f"   Ready for Training: {summary['ready_timeframes']}")
        print(f"   Ready List: {', '.join(summary['ready_list'])}")

        if summary["all_ready"]:
            print("   ✅ ALL TIMEFRAMES READY FOR TRAINING!")
        else:
            print("   ⚠️  SOME TIMEFRAMES NOT READY")

        # Detailed breakdown
        print("\n📋 DETAILED BREAKDOWN:")
        for timeframe in report["timeframes"]:
            data_status = report["data_files"][timeframe]
            model_status = report["model_files"][timeframe]
            completeness = report["completeness"][timeframe]

            status_icon = (
                "✅" if completeness.get("ready_for_training", False) else "❌"
            )
            print(f"\n   {status_icon} {timeframe}:")
            print(
                f"      Data: CSV={data_status['csv_exists']}, Parquet={data_status['parquet_exists']}",
            )
            print(
                f"      Models: Ensemble={model_status['ensemble_exists']}, HMM={model_status['hmm_exists']}",
            )
            print(f"      Ready: {completeness.get('ready_for_training', False)}")

        if "error" in completeness:
                print(f"      Error: {completeness['error']}")

        print("\n" + "=" * 80)

def main():
    """Main function to run the verification."""
    parser, argparse.ArgumentParser(
        description="Verify timeframe data for multi-timeframe HMM ensemble",
    )
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save report JSON",
    )

    args, parser.parse_args()

    if True:
        # Load configuration
        config = CONFIG if hasattr(CONFIG, "get") else {}

        # Create verifier
        verifier = TimeframeDataVerifier(config)

        # Generate report
        report = verifier.generate_report()

        # Print report
        verifier.print_report(report)

        # Save report if requested
        if args.output:
            pass
        with open(args.output, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Report saved to {args.output}")

        # Return success/failure
        success = report["summary"]["all_ready"]
        if success:
            logger.info("✅ All timeframes verified and ready for training!")
        else:
            logger.warning("⚠️  Some timeframes need attention before training")

        return success

    pass
        logger.exception(f"💥 Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
