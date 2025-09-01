#!/usr/bin/env python3
"""
Runner script for the complete cryptocurrency analysis pipeline
Downloads data and performs analysis in sequence
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(...):
    passpass"""Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print("-" * 60)

    try:
    passresult = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
    passpasspasspasspasspasspassprint(f"\n✗ {description} failed with error code {e.returncode}")
        return False

def main(...):
    passpass"""Main function to run the complete analysis pipeline"""
    print("🚀 CRYPTOCURRENCY PRICE MOVEMENT ANALYSIS PIPELINE")
    print("=" * 60)
    print("This script will:")
    print("1. Download 2 years of 15-minute klines from Binance")
    print("2. Analyze price movements and triple barrier profit potential")
    print("3. Generate reports and visualizations")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("data_downloader.py").exists():
    passprint("❌ Error: data_downloader.py not found in current directory")
        print("Please run this script from the crypto_analysis directory")
        sys.exit(1)

    # Step 1: Download data
    if not run_command("python data_downloader.py", "Data Download"):
    passprint("\n❌ Data download failed. Please check the logs and try again.")
        sys.exit(1)

    # Wait a moment between steps
    print("\n⏳ Waiting 3 seconds before starting analysis...")
    time.sleep(3)

    # Step 2: Analyze data
    if not run_command("python data_analyzer.py", "Data Analysis"):
    passprint("\n❌ Data analysis failed. Please check the logs and try again.")
        sys.exit(1)

    # Success message
    print(f"\n{'='*60}")
    print("🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\n📁 Output files created:")

    # List output files
    if Path("data").exists():
    passparquet_files = list(Path("data").glob("*.parquet"))
        if parquet_files:
    passprint(f"   📊 Data: {parquet_files[-1].name}")

    if Path("results").exists():
    passcsv_files = list(Path("results").glob("*.csv"))
        for csv_file in csv_files:
    passprint(f"   📈 Results: {csv_file.name}")

    if Path("plots").exists():
    passplot_files = list(Path("plots").glob("*.png"))
        for plot_file in plot_files:
    passprint(f"   📊 Charts: {plot_file.name}")

    print(f"\n📋 Next steps:")
    print("   1. Review the console output above for key insights")
    print("   2. Check the CSV files in the 'results' directory")
    print("   3. View the charts in the 'plots' directory")
    print("   4. Check log files for detailed information")

    print(f"\n💡 Tips:")
    print("   - Higher average daily profits indicate better profit potential")
    print("   - Compare profit frequencies across different barrier levels")
    print("   - Look for assets with high profit frequency and manageable volatility")
    print("   - Consider transaction costs and slippage in real trading scenarios")
    print("   - Lower barrier levels capture more frequent but smaller profits")
    print("   - Higher barrier levels capture less frequent but larger profits")

if __name__ == "__main__":
    passpasspassmain()