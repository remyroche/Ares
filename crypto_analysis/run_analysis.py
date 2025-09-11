#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Runner script for the complete cryptocurrency analysis pipeline
Downloads data and performs analysis in sequence
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    tprint(f"\n{'='*60}")
    tprint(f"STEP: {description}")
    tprint(f"{'='*60}")
    tprint(f"Running: {command}")
    tprint("-" * 60)

    try:
        subprocess.run(command, shell=True, check=True, capture_output=False)
        tprint(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        tprint(f"\n✗ {description} failed with error code {e.returncode}")
        return False

def main():
    """Main function to run the complete analysis pipeline"""
    tprint("🚀 CRYPTOCURRENCY PRICE MOVEMENT ANALYSIS PIPELINE")
    tprint("=" * 60)
    tprint("This script will:")
    tprint("1. Download 2 years of 15-minute klines from Binance")
    tprint("2. Analyze price movements and triple barrier profit potential")
    tprint("3. Generate reports and visualizations")
    tprint("=" * 60)

    # Check if we're in the right directory
    if not Path("data_downloader.py").exists():
        tprint("❌ Error: data_downloader.py not found in current directory")
        tprint("Please run this script from the crypto_analysis directory")
        sys.exit(1)

    # Step 1: Download data
    if not run_command("python data_downloader.py", "Data Download"):
        tprint("\n❌ Data download failed. Please check the logs and try again.")
        sys.exit(1)

    # Wait a moment between steps
    tprint("\n⏳ Waiting 3 seconds before starting analysis...")
    time.sleep(3)

    # Step 2: Analyze data
    if not run_command("python data_analyzer.py", "Data Analysis"):
        tprint("\n❌ Data analysis failed. Please check the logs and try again.")
        sys.exit(1)

    # Success message
    tprint(f"\n{'='*60}")
    tprint("🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    tprint(f"{'='*60}")
    tprint("\n📁 Output files created:")

    # List output files
    if Path("data").exists():
        parquet_files = list(Path("data").glob("*.parquet"))
        if parquet_files:
            tprint(f"   📊 Data: {parquet_files[-1].name}")

    if Path("results").exists():
        csv_files = list(Path("results").glob("*.csv"))
        for csv_file in csv_files:
            tprint(f"   📈 Results: {csv_file.name}")

    if Path("plots").exists():
        plot_files = list(Path("plots").glob("*.png"))
        for plot_file in plot_files:
            tprint(f"   📊 Charts: {plot_file.name}")

    tprint("\n📋 Next steps:")
    tprint("   1. Review the console output above for key insights")
    tprint("   2. Check the CSV files in the 'results' directory")
    tprint("   3. View the charts in the 'plots' directory")
    tprint("   4. Check log files for detailed information")

    tprint("\n💡 Tips:")
    tprint("   - Higher average daily profits indicate better profit potential")
    tprint("   - Compare profit frequencies across different barrier levels")
    tprint("   - Look for assets with high profit frequency and manageable volatility")
    tprint("   - Consider transaction costs and slippage in real trading scenarios")
    tprint("   - Lower barrier levels capture more frequent but smaller profits")
    tprint("   - Higher barrier levels capture less frequent but larger profits")

if __name__ == "__main__":
    await main()
