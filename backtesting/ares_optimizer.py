import pandas as pd
import numpy as np
import itertools
import os
import copy
import traceback
# Import the main CONFIG dictionary
from src.config import CONFIG
from backtesting.ares_data_preparer import load_raw_data, calculate_features_and_score, get_sr_levels 
from backtesting.ares_backtester import run_backtest 
from emails.ares_mailer import send_email

# --- Efficient Optimization Configuration ---

# The Coarse Grid is now expanded to test both weights AND key indicator values.
# Access OPTIMIZATION_CONFIG from CONFIG
COARSE_PARAM_GRID = CONFIG['backtesting']['optimization']['params']

NUM_TUNING_POINTS = CONFIG.get('fine_tune_num_points', 10)
TUNING_PERCENTAGE = CONFIG.get('fine_tune_ranges_multiplier', 0.2)
INTEGER_PARAMS = CONFIG.get('integer_params', ['adx_period', 'atr_period', 'sma_period'])

TOP_N_RESULTS = 5 # This can be moved to CONFIG if desired

def run_grid_search_stage(param_grid, klines_df, agg_trades_df, futures_df, sr_levels, stage_name):
    """Runs a full grid search for a given parameter grid."""
    print(f"\n--- Starting {stage_name} ---")
    
    # The param_grid here is already the COARSE_GRID_RANGES from CONFIG
    # We need to flatten the nested structure of param_grid for itertools.product
    # and then reconstruct the nested dictionary for each combination.
    
    # Convert nested grid to flat list of (path, values) pairs
    flat_param_paths = []
    flat_param_values = []

    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict) and 'type' in v: # It's a parameter definition
                 flat_param_paths.append(new_key)
                 if v['type'] == 'int':
                     flat_param_values.append(np.linspace(v['low'], v['high'], 5, dtype=int)) # 5 steps for coarse grid
                 elif v['type'] == 'float':
                     flat_param_values.append(np.linspace(v['low'], v['high'], 5))
                 elif v['type'] == 'categorical':
                     flat_param_values.append(v['choices'])
            elif isinstance(v, dict):
                flatten_dict(v, new_key)

    flatten_dict(param_grid)

    keys = flat_param_paths
    values = flat_param_values
    
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(param_combinations)} combinations for this stage...")
    
    results = []
    for i, flat_params_combo in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {flat_params_combo}")
        
        # Convert flat_params_combo back to nested structure for calculate_features_and_score
        current_params = copy.deepcopy(CONFIG['best_params'])
        for path, value in flat_params_combo.items():
            set_param_value_at_path(current_params, path.split('.'), value)
        
        # Pass sr_levels to calculate_features_and_score
        prepared_df = calculate_features_and_score(
            klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), current_params, sr_levels
        )
        
        portfolio_result = run_backtest(prepared_df, current_params) # Pass current_params to backtest
        
        if len(portfolio_result.trades) == 0:
            print("--> NO TRADES WERE MADE.")
        else:
            print(f"--> Resulting Equity: ${portfolio_result.equity:,.2f} | Total Trades: {len(portfolio_result.trades)}")
        
        results.append({'params': current_params, 'portfolio': portfolio_result})
        
    results.sort(key=lambda x: x['portfolio'].equity, reverse=True)
    return results

def run_coordinate_descent_stage(best_params, klines_df, agg_trades_df, futures_df, sr_levels):
    """Performs Stage 2: Fine-tunes each parameter one by one."""
    print("\n--- Starting Stage 2: Coordinate Descent Fine-Tuning ---")
    
    current_best_params = copy.deepcopy(best_params)
    
    all_param_paths = []
    def collect_paths(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict) and 'type' in v:
                all_param_paths.append(new_key)
            elif isinstance(v, dict):
                collect_paths(v, new_key)
    
    collect_paths(CONFIG['backtesting']['optimization']['params'])

    for param_to_tune in all_param_paths:
        print(f"\n--- Tuning Parameter: '{param_to_tune}' ---")
        
        parts = param_to_tune.split('.')
        best_value = get_param_value_from_path(current_best_params, parts)

        if best_value is None:
            print(f"Warning: Parameter '{param_to_tune}' not found in current best params. Skipping tuning.")
            continue

        variation = abs(best_value * TUNING_PERCENTAGE)
        low_val = best_value - variation
        high_val = best_value + variation
        
        tuning_values = np.linspace(low_val, high_val, num=NUM_TUNING_POINTS)
        if param_to_tune in INTEGER_PARAMS:
            tuning_values = np.unique(np.round(tuning_values)).astype(int)
        
        print(f"Testing values: {tuning_values}")
        
        best_equity_for_param = -np.inf
        best_value_for_param = None

        for value in tuning_values:
            test_params = copy.deepcopy(current_best_params)
            set_param_value_at_path(test_params, parts, value)

            prepared_df = calculate_features_and_score(
                klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), test_params, sr_levels
            )
            
            portfolio_result = run_backtest(prepared_df, test_params)
            
            print(f"  Value: {value:<10} -> Equity: ${portfolio_result.equity:,.2f}")

            if portfolio_result.equity > best_equity_for_param:
                best_equity_for_param = portfolio_result.equity
                best_value_for_param = value
        
        if best_value_for_param is not None:
            print(f"==> Best value for '{param_to_tune}' is {best_value_for_param}. Updating parameters.")
            set_param_value_at_path(current_best_params, parts, best_value_for_param)

    print("\n--- Running Final Backtest with Tuned Parameters ---")
    prepared_df = calculate_features_and_score(
        klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), current_best_params, sr_levels
    )
    final_portfolio = run_backtest(prepared_df, current_best_params)

    return [{'params': current_best_params, 'portfolio': final_portfolio}]

def format_report_to_string(final_results):
    report_lines = []
    separator = "="*80
    report_lines.append("Optimization Complete: Final Tuned Parameters")
    report_lines.append(separator)
    for i, result in enumerate(final_results[:TOP_N_RESULTS]):
        portfolio = result['portfolio']
        params = result['params']
        report_lines.append(f"\n--- Rank #{i+1} ---")
        report_lines.append(f"Final Equity: ${portfolio.equity:,.2f}")
        report_lines.append(f"Total Trades: {len(portfolio.trades)}")
        report_lines.append("Parameters:")
        report_lines.append(str(params))
        report = portfolio.report()
        if not report:
            report_lines.append("No trades were made with this parameter set.")
            continue
        sorted_report = sorted(report.items(), key=lambda item: item[1].get('num_trades', 0), reverse=True)
        report_lines.append(f"\n{'Trade Source':<25} | {'Num Trades':>12} | {'Sharpe Ratio':>15} | {'Avg Profit/Loss (%)':>20}")
        report_lines.append("-" * 80)
        for source, metrics in sorted_report:
            report_lines.append(f"{source:<25} | {metrics['num_trades']:>12} | {metrics['sharpe']:>15.2f} | {metrics['avg_profit_pct']:>20.2f}%")
    report_lines.append("\n" + separator)
    report_lines.append("RECOMMENDATION: Update the best_params dictionary in config.py with the final tuned values.")
    report_lines.append(separator)
    return "\n".join(report_lines)

# Helper functions for nested dictionary access
def get_param_value_from_path(base_dict, path_parts):
    """Helper to get a nested parameter value from a dictionary using a list of path parts."""
    current = base_dict
    for part in path_parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None # Path not found
    return current

def set_param_value_at_path(base_dict, path_parts, value):
    """Helper to set a nested parameter value in a dictionary using a list of path parts."""
    current = base_dict
    for i, part in enumerate(path_parts):
        if i == len(path_parts) - 1:
            current[part] = value
        else:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part] # CORRECTED: Changed 'p' to 'part'

# This main block is now for running the optimizer as a standalone script
if __name__ == "__main__":
    email_subject = "Ares Optimizer FAILED"
    email_body = "The script encountered an unexpected error."
    try:
        print("--- Ares Efficient Strategy Optimizer (Standalone) ---")
        klines_df, agg_trades_df, futures_df = load_raw_data() # Load futures_df here
        daily_df = klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        sr_levels = get_sr_levels(daily_df)
        
        coarse_results = run_grid_search_stage(COARSE_PARAM_GRID, klines_df, agg_trades_df, futures_df, sr_levels, "Stage 1: Coarse Grid Search")
        
        initial_equity = CONFIG['initial_equity']

        if not coarse_results or coarse_results[0]['portfolio'].equity <= initial_equity:
            print("\nCoarse search did not yield any profitable results. Exiting.")
            email_subject = "Ares Optimizer Finished (No Profitable Results)"
            email_body = "The coarse search stage did not find any parameter combinations that resulted in a profit."
        else:
            best_coarse_params = coarse_results[0]['params']
            print(f"\nBest result from Coarse Search: ${coarse_results[0]['portfolio'].equity:,.2f}")
            print(f"Best Coarse Params: {best_coarse_params}")
            final_results = run_coordinate_descent_stage(best_coarse_params, klines_df, agg_trades_df, futures_df, sr_levels)
            report_string = format_report_to_string(final_results)
            print("\n" + report_string)
            email_subject = f"Ares Optimizer Complete - Best Equity: ${final_results[0]['portfolio'].equity:,.2f}"
            email_body = report_string
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        email_body = f"An exception occurred during the optimization process:\n\n{traceback.format_exc()}"
        raise
    finally:
        send_email(email_subject, email_body)
