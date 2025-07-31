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
    run_monte_carlo_simulation,
)
from backtesting.ares_data_preparer import calculate_and_label_regimes, get_sr_levels
from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG  # Import CONFIG
from src.utils.error_handler import handle_errors


@handle_errors(
    exceptions=(Exception,), default_return=False, context="monte_carlo_validation_step"
)
async def run_step(symbol: str, data_dir: str) -> bool:
    """
    Runs Monte Carlo validation using the fully prepared data from Step 4.
    Loads prepared data from the pickle file saved by step4.
    """
    setup_logging()
    logger = system_logger.getChild("Step7MonteCarloValidation")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 7: MONTE CARLO VALIDATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"📁 Data directory: {data_dir}")

    try:
        # Step 7.1: Load PREPARED data from the file saved in step 4
        logger.info("📊 STEP 7.1: Loading prepared data from pickle file...")
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
            logger.error("❌ Prepared data is empty for Monte Carlo validation. Aborting.")
            return False

        # Step 7.2: Load best parameters
        logger.info("🎯 STEP 7.2: Loading best parameters...")
        params_load_start = time.time()
        
        best_params = CONFIG.get("best_params", {})  # Use the globally updated best_params
        
        params_load_duration = time.time() - params_load_start
        logger.info(f"⏱️  Parameters loading completed in {params_load_duration:.2f} seconds")
        logger.info(f"✅ Loaded best parameters:")
        for param, value in best_params.items():
            logger.info(f"   - {param}: {value}")

        # Step 7.3: Run Monte Carlo simulation
        logger.info("🎲 STEP 7.3: Running Monte Carlo simulation...")
        mc_start = time.time()
        
        logger.info(f"🔢 Monte Carlo simulation parameters:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Data shape: {prepared_data.shape}")
        logger.info(f"   - Best params count: {len(best_params)}")
        
        mc_curves, base_portfolio, mc_report_str = run_monte_carlo_simulation(
            prepared_data, best_params
        )
        
        mc_duration = time.time() - mc_start
        logger.info(f"⏱️  Monte Carlo simulation completed in {mc_duration:.2f} seconds")
        logger.info(f"📊 Simulation results:")
        logger.info(f"   - Number of curves: {len(mc_curves) if mc_curves is not None else 0}")
        logger.info(f"   - Base portfolio present: {base_portfolio is not None}")
        logger.info(f"   - Report length: {len(mc_report_str)} characters")

        # Step 7.4: Save reports
        logger.info("💾 STEP 7.4: Saving reports...")
        save_start = time.time()
        
        reports_dir = os.path.join(
            Path(__file__).parent.parent.parent, "reports"
        )  # Access reports dir
        os.makedirs(reports_dir, exist_ok=True)
        mc_file = os.path.join(reports_dir, f"{symbol}_monte_carlo_report.txt")
        
        with open(mc_file, "w") as f:
            f.write(mc_report_str)

        # Parse metrics from the report string
        metrics = _parse_report_for_metrics(mc_report_str)

        # Save metrics to a JSON file for the TrainingManager
        output_metrics_file = os.path.join(data_dir, f"{symbol}_mc_metrics.json")
        with open(output_metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Reports saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Reports saved to:")
        logger.info(f"   - Monte Carlo report: {mc_file}")
        logger.info(f"   - Metrics JSON: {output_metrics_file}")
        logger.info(f"📊 Extracted metrics:")
        for metric, value in metrics.items():
            logger.info(f"   - {metric}: {value}")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 7: MONTE CARLO VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📄 Output files:")
        logger.info(f"   - Monte Carlo report: {mc_file}")
        logger.info(f"   - Metrics JSON: {output_metrics_file}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 7: MONTE CARLO VALIDATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


def _parse_report_for_metrics(report_content: str) -> Dict[str, float]:
    """Parses a text report to extract key-value metrics."""
    import re

    metrics = {}
    # Adjusted patterns to match the output format of calculate_detailed_metrics and run_monte_carlo_simulation
    patterns = {
        "Original Final Equity": r"Original Final Equity:\s*\$([0-9,]+\.\d{2})",
        "Mean Simulated Equity": r"Mean Simulated Equity:\s*\$([0-9,]+\.\d{2})",
        "Confidence Interval Lower": r"Confidence Interval for Final Equity: \$([0-9,]+\.\d{2})",
        "Confidence Interval Upper": r"Confidence Interval for Final Equity: \$[0-9,]+\.\d{2} - \$([0-9,]+\.\d{2})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_content)
        if match:
            try:
                # For ranges, we might need to extract both groups
                if "Confidence Interval" in key:
                    if "Lower" in key:
                        value = match.group(1).replace(",", "")
                    else:  # Upper
                        value = (
                            match.group(2).replace(",", "") if match.groups() else ""
                        )  # Ensure group exists
                else:
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
