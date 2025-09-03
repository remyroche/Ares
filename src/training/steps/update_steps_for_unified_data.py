# src/training/steps/update_steps_for_unified_data.py

"""Utility script to update all training steps to use the unified data loader."

This script provides guidance and templates for updating the training steps
to use the new unified Parquet partitioned data format.
"""

from __future__ import annotations

from typing import Any, Dict, List

# List of all training steps that need to be updated
TRAINING_STEPS: List[str] = [
    "step2_market_regime_classification",
    "step3_regime_data_splitting",
    "step4_analyst_labeling_feature_engineering",
    "step5_hmm_based_training",
    "step6_analyst_enhancement",
    "step7_analyst_ensemble_creation",
    "step8_tactician_labeling",
    "step9_tactician_specialist_training",
    "step10_tactician_ensemble_creation",
    "step11_confidence_calibration",
    "step17_final_parameters_optimization",
    "step13_walk_forward_validation",
    "step14_monte_carlo_validation",
    "step15_ab_testing",
    "step16_saving",
]


def get_unified_data_loader_import() -> str:
    """Get the import statement for the unified data loader."""
    return (
        "from src.training.steps.unified_data_loader import get_unified_data_loader"
    )


def get_unified_data_loading_code(
    symbol_var: str = "symbol",
    exchange_var: str = "exchange",
    timeframe_var: str = "timeframe",
    lookback_days: int = 180,
    data_dir_var: str = "data_dir",
) -> str:
    """Get the code template for loading unified data."

    Args:
        symbol_var: Variable name for symbol
        exchange_var: Variable name for exchange
        timeframe_var: Variable name for timeframe
        lookback_days: Number of days to look back
        data_dir_var: Variable name for data directory

    Returns:
        Code template string

    """
    return f"""
        # Use unified data loader to get data
        self.logger.info("🔄 Loading data using unified data loader...")
        data_loader = get_unified_data_loader(self.config)

        # Load unified data
        historical_data = await data_loader.load_unified_data(
            symbol={symbol_var},
            exchange={exchange_var},
            timeframe={timeframe_var},
            lookback_days={lookback_days},
            data_dir={data_dir_var},
        )

        if historical_data is None or historical_data.empty:
            self.logger.error("❌ No data found - check symbol and exchange configuration")
            raise ValueError(f"No data found for {{symbol}} on {{exchange}}")

        # Log data information
        data_info = data_loader.get_data_info(historical_data)
        self.logger.info(f"✅ Loaded unified data: {{data_info['rows']}} rows")
        self.logger.info(f"   Date range: {{data_info['date_range']['start']}} to {{data_info['date_range']['end']}}")
        self.logger.info(f"   Has aggtrades data: {{data_info['has_aggtrades_data']}}")
        self.logger.info(f"   Has futures data: {{data_info['has_futures_data']}}")

        # Ensure we have the required OHLCV columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            self.logger.error(f"❌ Missing required columns: {{missing_columns}}")
            raise ValueError(f"Missing required columns: {{missing_columns}}")
    """


def get_step_specific_guidance(step_name: str) -> Dict[str, Any]:
    """Get step-specific guidance for updating."""
    from src.config.constants import (
        BLANK_TRAINING_LOOKBACK_DAYS,
    )

    # High complexity areas that need special attention

    guidance: Dict[str, Any] = {
        "step2_market_regime_classification": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1h",  # Regime classification typically uses 1h
            "notes": "May need to resample data to 1h timeframe for regime classification",
        },
        "step3_regime_data_splitting": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Uses regime classification results from step02",
        },
        "step4_analyst_labeling_feature_engineering": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Needs both OHLCV data and regime labels",
        },
        "step5_hmm_based_training": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Uses labeled data from step04",
        },
        "step6_analyst_enhancement": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Uses trained models from step05",
        },
        "step7_analyst_ensemble_creation": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Creates ensemble from step06 models",
        },
        "step8_tactician_labeling": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Needs both OHLCV data and analyst predictions",
        },
        "step9_tactician_specialist_training": {
            "lookback_days": BLANK_TRAINING_LOOKBACK_DAYS,
            "timeframe": "1m",
            "notes": "Uses labeled data from step08",
        },
        "step10_tactician_ensemble_creation": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Creates ensemble from step09 models",
        },
        "step11_confidence_calibration": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Uses predictions from step10",
        },
        "step17_final_parameters_optimization": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Optimizes parameters using all previous results",
        },
        "step13_walk_forward_validation": {
            "lookback_days": 365,  # Longer period for validation
            "timeframe": "1m",
            "notes": "Performs walk-forward validation",
        },
        "step14_monte_carlo_validation": {
            "lookback_days": 365,
            "timeframe": "1m",
            "notes": "Performs Monte Carlo validation",
        },
        "step15_ab_testing": {
            "lookback_days": 90,  # Shorter period for A/B testing
            "timeframe": "1m",
            "notes": "Performs A/B testing",
        },
        "step16_saving": {
            "lookback_days": 30,  # Minimal data needed for saving
            "timeframe": "1m",
            "notes": "Saves results and models",
        },
    }

    return guidance.get(
        step_name,
        {"lookback_days": 180, "timeframe": "1m", "notes": "Standard data loading"},
    )


def generate_step_update_template(step_name: str) -> str:
    """Generate a template for updating a specific step."""
    guidance = get_step_specific_guidance(step_name)

    return f"""
# Template for updating {step_name}.py

## 1. Add import at the top of the file:
{get_unified_data_loader_import()}

## 2. Replace existing data loading code with:
{get_unified_data_loading_code(
    lookback_days=guidance['lookback_days'],
    timeframe_var=f'\"{guidance["timeframe"]}\"',
)}

## 3. Step-specific considerations:
# {guidance['notes']}

## 4. Additional data processing (if needed):
# - If the step needs regime labels, load them from step02 results
# - If the step needs analyst predictions, load them from step07 results
# - If the step needs tactician predictions, load them from step10 results

## 5. Example of loading additional data:
# regime_file_path = f"{{data_dir}}/{{exchange}}_{{symbol}}_regime_classification.json"
# if os.path.exists(regime_file_path):
#     with open(regime_file_path, 'r') as f:
#         regime_data = json.load(f)
#     # Process regime data as needed
"""



def main() -> None:
    """Main function to generate update guidance."""
    high_complexity_areas = {
        "step1_data_collection": "❌ HIGH COMPLEXITY - consolidate_files (D-23), run_step (C-18)",
        "step4_main_model_training": "❌ HIGH COMPLEXITY - run_step (C-13)",
        "step5_multi_stage_hpo": "⚠️  MEDIUM COMPLEXITY - run_step (B-9)",
        "step7_monte_carlo_validation": "⚠️  MEDIUM COMPLEXITY - run_step (B-7)",
        "step6_walk_forward_validation": "⚠️  MEDIUM COMPLEXITY - run_step (B-6)",
        "step9_save_results": "⚠️  MEDIUM COMPLEXITY - run_step (B-6)",
        "step3_coarse_optimization": "⚠️  MEDIUM COMPLEXITY - run_step (B-6)",
        "step2_preliminary_optimization": "✅ LOW COMPLEXITY - run_step (A-5)",
        "step8_ab_testing_setup": "✅ LOW COMPLEXITY - run_step (A-2)",
    }

    for i, step in enumerate(TRAINING_STEPS, 1):
        _ = i  # preserved for clarity; index may be used later
        guidance = get_step_specific_guidance(step)
        _ = guidance  # ensure call side effects are preserved if any

        if step in high_complexity_areas:
            # Here we would log or highlight complexity areas for the developer
            pass

        # Generate template (could be written to disk or printed)
        template = generate_step_update_template(step)
        print(template)


if __name__ == "__main__":
    await main()