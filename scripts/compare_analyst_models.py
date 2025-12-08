"""
Script to compare different Analyst Base models.

This script automates the training, gating, optimization, and backtesting of multiple
Analyst Base model types (e.g., 'lgbm', 'ngboost', 'bayesianridge').

Workflow per model:
1.  Configure `analyst_base_config.yaml` to enable ONLY the specific model.
2.  Run `UnifiedModelsTrainingStep` (via `AnalystBaseTrainingStep`).
3.  Run `GateTrainingStep`.
4.  Run `MetaLabelingHPOExperimentStep` (to optimize TP/SL/Trailing params).
5.  Run `AnalystBaseBacktestStep` (to evaluate performance).
6.  Collect results.

Finally, generates a comparison summary.

Usage:
    python scripts/compare_analyst_models.py --symbol ETHUSDT --models lgbm ngboost
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.getcwd())

from src.utils.logger import setup_unified_logging, get_unified_logger, get_system_logger
from src.utils.tprint import tprint, tprint_success, tprint_error, tprint_info

# Import Steps
from src.utils.logger import setup_unified_logging, get_unified_logger, get_system_logger
from src.training.steps.model_training.analyst_base_training_step import AnalystBaseTrainingStep
from src.training.steps.model_training.gate_training_step import GateTrainingStep
from src.training.steps.labeling.meta_labeling_hpo_experiment_step import MetaLabelingHPOExperimentStep
from src.training.steps.backtesting.analyst_base_backtest_step import AnalystBaseBacktestStep

# Constants
CONFIG_PATH = Path("src/training/steps/model_training/analyst_base_config.yaml")
RESULTS_DIR = Path("outcomes/model_comparison")
system_logger = get_system_logger()

async def run_comparison(symbol: str, exchange: str, timeframe: str, direction: str, models: List[str]):
    """Run comparison workflow."""

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_results = []

    # Read original config to backup
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            original_config_content = f.read()
            original_config = yaml.safe_load(original_config_content)
    else:
        tprint_error(f"Config file not found: {CONFIG_PATH}")
        return

    try:
        for model_name in models:
            tprint("=" * 80)
            tprint(f"🚀 Processing Model: {model_name.upper()}", "header")
            tprint("=" * 80)

            # 1. Modify Config
            tprint_info(f"⚙️ Configuring {model_name}...")
            current_config = yaml.safe_load(original_config_content) # Reload fresh

            base_models = current_config.get("analyst_config", {}).get("base_models", {})

            if model_name not in base_models:
                tprint_error(f"Model {model_name} not found in config. Skipping.")
                continue

            # Disable all, enable target
            for m in base_models:
                base_models[m]["enabled"] = (m == model_name)

            # Save modified config
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(current_config, f)

            step_config = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "direction": direction,
                "execution_mode": "full" # or light
            }

            # 2. Train Base Model
            tprint_info("🏋️ Training Base Model...")
            train_step = AnalystBaseTrainingStep()
            train_result = await train_step.run(step_config)

            if not train_result.get("success"):
                tprint_error(f"Training failed for {model_name}")
                continue

            # 3. Train Gate Model
            tprint_info("⛩️ Training Gate Model...")
            gate_step = GateTrainingStep()
            gate_result = await gate_step.run(step_config)

            if not gate_result.get("success"):
                tprint_error(f"Gate training failed for {model_name}")
                # Continue? Gate is optional for backtest but important for comparison

            # 4. Meta Labeling HPO (Best Params)
            tprint_info("🎯 Optimizing TP/SL/Trailing (Meta Labeling HPO)...")
            hpo_step = MetaLabelingHPOExperimentStep()
            hpo_result = await hpo_step.run(step_config)

            if not hpo_result.get("success"):
                tprint_error(f"HPO failed for {model_name}. Using default params.")

            # 5. Backtest
            tprint_info("🧪 Backtesting...")
            backtest_step = AnalystBaseBacktestStep()
            backtest_result = await backtest_step.run(step_config)

            if not backtest_result.get("success"):
                tprint_error(f"Backtest failed for {model_name}")
                continue

            # 6. Collect Results
            metrics = backtest_result.get("metrics", {})
            report_path = backtest_result.get("artifacts", {}).get("backtest_report_markdown")

            if report_path:
                dest_path = RESULTS_DIR / f"{model_name}_{Path(report_path).name}"
                shutil.copy(report_path, dest_path)
                tprint_success(f"Report saved to {dest_path}")

            summary_results.append({
                "Model": model_name,
                "Sharpe": metrics.get("sharpe_ratio", 0.0),
                "Gated Sharpe": metrics.get("gated_sharpe_ratio", "N/A"),
                "Total Return": metrics.get("total_return", 0.0),
                "Win Rate": metrics.get("win_rate", 0.0),
                "Trades": metrics.get("approx_trades", 0)
            })

            tprint_success(f"✅ Completed {model_name}")

    finally:
        # Restore original config
        tprint_info("Restoring original config...")
        with open(CONFIG_PATH, "w") as f:
            f.write(original_config_content)

    # Generate Summary
    tprint("=" * 80)
    tprint("📊 COMPARISON SUMMARY", "header")
    tprint("=" * 80)

    print(f"{'Model':<15} {'Sharpe':<10} {'Gated Sharpe':<15} {'Return':<10} {'Win Rate':<10} {'Trades':<10}")
    print("-" * 75)
    for res in summary_results:
        gated_s = f"{res['Gated Sharpe']:.4f}" if isinstance(res['Gated Sharpe'], float) else str(res['Gated Sharpe'])
        print(f"{res['Model']:<15} {res['Sharpe']:<10.4f} {gated_s:<15} {res['Total Return']:<10.4%} {res['Win Rate']:<10.2%} {res['Trades']:<10}")

    # Find Best
    if summary_results:
        # Sort by Sharpe (or Gated Sharpe if available and preferred)
        # Here we sort by raw Sharpe for base comparison
        summary_results.sort(key=lambda x: x["Sharpe"], reverse=True)
        best_model = summary_results[0]

        tprint("=" * 80)
        tprint(f"🏆 Best Model: {best_model['Model'].upper()}", "success")
        tprint(f"   Sharpe: {best_model['Sharpe']:.4f}")
        tprint("=" * 80)

        tprint_info("Next Steps:")
        tprint_info(f"1. Update 'analyst_base_config.yaml' to enable ONLY '{best_model['Model']}'.")
        tprint_info("2. Run 'FinalParametersOptimizer' to fine-tune the system.")

def main():
    parser = argparse.ArgumentParser(description="Compare Analyst Base Models")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--models", nargs="+", default=["lgbm", "ngboost", "bayesianridge"], help="List of models to compare")

    args = parser.parse_args()

    # Setup logger
    setup_unified_logging()

    asyncio.run(run_comparison(args.symbol, args.exchange, args.timeframe, args.direction, args.models))

if __name__ == "__main__":
    main()
