# src/training/steps/update_steps_for_unified_data.py

"""
Utility script to update all training steps to use the unified data loader.

This script provides guidance and templates for updating the training steps
to use the new unified Parquet partitioned data format.
"""

import os
from typing import List, Dict, Any

# List of all training steps that need to be updated
TRAINING_STEPS = [
    "step2_market_regime_classification",
    "step3_regime_data_splitting",
    "step4_analyst_labeling_feature_engineering",
    "step5_analyst_specialist_training",
    "step6_analyst_enhancement",
    "step7_analyst_ensemble_creation",
    "step8_tactician_labeling",
    "step9_tactician_specialist_training",
    "step10_tactician_ensemble_creation",
    "step11_confidence_calibration",
    "step12_final_parameters_optimization",
    "step13_walk_forward_validation",
    "step14_monte_carlo_validation",
    "step15_ab_testing",
    "step16_saving",
]


def get_unified_data_loader_import() -> str:
    """Get the import statement for the unified data loader."""
    return (
        """from src.training.steps.unified_data_loader import get_unified_data_loader"""
    )


def get_unified_data_loading_code(
    symbol_var: str = "symbol",
    exchange_var: str = "exchange",
    timeframe_var: str = "timeframe",
    lookback_days: int = 180,
    data_dir_var: str = "data_dir",
) -> str:
    """
    Get the code template for loading unified data.

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
            lookback_days={lookback_days}
        )
        
        if historical_data is None or historical_data.empty:
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
            raise ValueError(f"Missing required columns: {{missing_columns}}")
"""


def get_step_specific_guidance(step_name: str) -> Dict[str, Any]:
    """Get step-specific guidance for updating."""
    guidance = {
        "step2_market_regime_classification": {
            "lookback_days": 180,
            "timeframe": "1h",  # Regime classification typically uses 1h
            "notes": "May need to resample data to 1h timeframe for regime classification",
        },
        "step3_regime_data_splitting": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Uses regime classification results from step2",
        },
        "step4_analyst_labeling_feature_engineering": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Needs both OHLCV data and regime labels",
        },
        "step5_analyst_specialist_training": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Uses labeled data from step4",
        },
        "step6_analyst_enhancement": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Uses trained models from step5",
        },
        "step7_analyst_ensemble_creation": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Creates ensemble from step6 models",
        },
        "step8_tactician_labeling": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Needs both OHLCV data and analyst predictions",
        },
        "step9_tactician_specialist_training": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Uses labeled data from step8",
        },
        "step10_tactician_ensemble_creation": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Creates ensemble from step9 models",
        },
        "step11_confidence_calibration": {
            "lookback_days": 180,
            "timeframe": "1m",
            "notes": "Uses predictions from step10",
        },
        "step12_final_parameters_optimization": {
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

    template = f"""
# Template for updating {step_name}.py

## 1. Add import at the top of the file:
{get_unified_data_loader_import()}

## 2. Replace existing data loading code with:
{get_unified_data_loading_code(
    lookback_days=guidance['lookback_days'],
    timeframe_var=f'"{guidance["timeframe"]}"'
)}

## 3. Step-specific considerations:
# {guidance['notes']}

## 4. Additional data processing (if needed):
# - If the step needs regime labels, load them from step2 results
# - If the step needs analyst predictions, load them from step7 results  
# - If the step needs tactician predictions, load them from step10 results

## 5. Example of loading additional data:
# regime_file_path = f"{data_dir}/{exchange}_{symbol}_regime_classification.json"
# if os.path.exists(regime_file_path):
#     with open(regime_file_path, 'r') as f:
#         regime_data = json.load(f)
#     # Process regime data as needed
"""

    return template


def main():
    """Main function to generate update guidance."""
    print("=" * 80)
    print("UNIFIED DATA LOADER UPDATE GUIDE")
    print("=" * 80)
    print()
    print("This guide helps update all training steps to use the unified data loader.")
    print("The unified data loader provides access to the new Parquet partitioned")
    print(
        "data format that includes klines, aggtrades, and futures data merged together."
    )
    print()

    print("STEPS TO UPDATE:")
    for i, step in enumerate(TRAINING_STEPS, 1):
        print(f"{i:2d}. {step}")
    print()

    print("GENERAL UPDATE PROCESS:")
    print("1. Add the unified data loader import")
    print("2. Replace existing data loading code with unified data loader calls")
    print("3. Update any step-specific data processing")
    print("4. Test the updated step")
    print()

    print("STEP-SPECIFIC TEMPLATES:")
    print("=" * 80)

    for step in TRAINING_STEPS:
        print(f"\n{step.upper()}:")
        print("-" * 40)
        guidance = get_step_specific_guidance(step)
        print(f"Lookback days: {guidance['lookback_days']}")
        print(f"Timeframe: {guidance['timeframe']}")
        print(f"Notes: {guidance['notes']}")

        # Generate template
        template = generate_step_update_template(step)
        print("\nTemplate:")
        print(template)
        print("=" * 80)


if __name__ == "__main__":
    main()
