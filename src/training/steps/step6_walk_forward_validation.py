import asyncio
import os
import sys
import pickle
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backtesting.ares_deep_analyzer import (
    run_walk_forward_analysis,
)
from backtesting.ares_data_preparer import calculate_and_label_regimes, get_sr_levels
from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG  # Import CONFIG
from src.utils.error_handler import handle_errors


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="walk_forward_validation_step",
)
async def run_step(symbol: str, data_dir: str) -> bool:
    """
    Runs walk-forward validation.
    Loads data from the specified pickle file.
    """
    setup_logging()
    logger = system_logger.getChild("Step6WalkForwardValidation")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 6: WALK-FORWARD VALIDATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"📁 Data directory: {data_dir}")

    try:
        # Step 6.1: Load PREPARED data from the file saved in step 4
        logger.info("📊 STEP 6.1: Loading data from pickle file...")
        data_load_start = time.time()
        
        prepared_data_path = os.path.join(data_dir, f"{symbol}_prepared_data.pkl")
        if not os.path.exists(prepared_data_path):
            logger.error(f"❌ Prepared data file not found: {prepared_data_path}")
            return False
            
        with open(prepared_data_path, "rb") as f:
            prepared_data = pickle.load(f)
        
        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        logger.info(f"📊 Loaded prepared data: {len(prepared_data)} rows, {len(prepared_data.columns)} columns")

        if prepared_data.empty:
            logger.error("❌ Prepared data is empty. Aborting.")
            return False

        # Step 6.2: Load best parameters
        logger.info("🎯 STEP 6.4: Loading best parameters...")
        params_load_start = time.time()
        
        best_params = CONFIG.get("best_params", {})  # Use the globally updated best_params
        
        params_load_duration = time.time() - params_load_start
        logger.info(f"⏱️  Parameters loading completed in {params_load_duration:.2f} seconds")
        logger.info(f"✅ Loaded best parameters:")
        for param, value in best_params.items():
            logger.info(f"   - {param}: {value}")

        # Step 6.3: Run walk-forward analysis
        logger.info("🔄 STEP 6.6: Running walk-forward analysis...")
        wfa_start = time.time()
        
        logger.info(f"🔢 Walk-forward analysis parameters:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Data shape: {prepared_data.shape}")
        logger.info(f"   - Best params count: {len(best_params)}")
        
        wfa_report_str = run_walk_forward_analysis(prepared_data, best_params)
        
        wfa_duration = time.time() - wfa_start
        logger.info(f"⏱️  Walk-forward analysis completed in {wfa_duration:.2f} seconds")
        logger.info(f"📊 Analysis report length: {len(wfa_report_str)} characters")

        # Step 6.4: Save reports
        logger.info("💾 STEP 6.7: Saving reports...")
        save_start = time.time()
        
        reports_dir = os.path.join(
            Path(__file__).parent.parent.parent, "reports"
        )  # Access reports dir
        os.makedirs(reports_dir, exist_ok=True)
        wfa_file = os.path.join(reports_dir, f"{symbol}_walk_forward_report.txt")
        
        with open(wfa_file, "w") as f:
            f.write(wfa_report_str)

        # Parse metrics from the report string
        metrics = _parse_report_for_metrics(wfa_report_str)

        # Save metrics to a JSON file for the TrainingManager
        output_metrics_file = os.path.join(data_dir, f"{symbol}_wfa_metrics.json")
        with open(output_metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Reports saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Reports saved to:")
        logger.info(f"   - Walk-forward report: {wfa_file}")
        logger.info(f"   - Metrics JSON: {output_metrics_file}")
        logger.info(f"📊 Extracted metrics:")
        for metric, value in metrics.items():
            logger.info(f"   - {metric}: {value}")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 6: WALK-FORWARD VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📄 Output files:")
        logger.info(f"   - Walk-forward report: {wfa_file}")
        logger.info(f"   - Metrics JSON: {output_metrics_file}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 6: WALK-FORWARD VALIDATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


def _parse_report_for_metrics(report_content: str) -> Dict[str, float]:
    """Parses a text report to extract key-value metrics."""
    import re

    metrics = {}
    # Adjusted patterns to match the output format of calculate_detailed_metrics and run_walk_forward_analysis
    patterns = {
        "Final Equity": r"Final Equity:\s*\$([0-9,]+\.\d{2})",
        "Total Trades": r"Total Trades:\s*(\d+)",
        "Sharpe Ratio": r"Sharpe Ratio:\s*(-?\d+\.\d{2})",
        "Sortino Ratio": r"Sortino Ratio:\s*(-?\d+\.\d{2})",
        "Max Drawdown (%)": r"Max Drawdown \(%\):\s*(-?\d+\.\d{2})",
        "Calmar Ratio": r"Calmar Ratio:\s*(-?\d+\.\d{2})",
        "Win Rate (%)": r"Win Rate \(%\):\s*(\d+\.\d{2})",
        "Profit Factor": r"Profit Factor:\s*(\d+\.\d{2})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_content)
        if match:
            try:
                # Remove commas from numbers if present (e.g., $10,000.00)
                value = match.group(1).replace(",", "")
                metrics[key.replace(" ", "_").replace("%", "Pct").replace(".", "")] = (
                    float(value)
                )
            except (ValueError, IndexError):
                continue
    return metrics


if __name__ == "__main__":
    # Command-line arguments: symbol, data_dir
    symbol = sys.argv[1]
    data_dir = sys.argv[2]

    success = asyncio.run(run_step(symbol, data_dir))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
