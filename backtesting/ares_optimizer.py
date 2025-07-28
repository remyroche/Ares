import pandas as pd
import numpy as np
import itertools
import os
import copy
import traceback
from config import (
    INITIAL_EQUITY
)
from ares_data_preparer import load_raw_data, calculate_features_and_score, get_sr_levels 
from ares_backtester import run_backtest 
from ares_mailer import send_email

# --- Efficient Optimization Configuration ---

# The Coarse Grid is now expanded to test both weights AND key indicator values.
COARSE_PARAM_GRID = {
    # --- Confidence Score Weights ---
    'weight_trend': [0.2, 0.5, 0.8], # Wider range for weights
    'weight_reversion': [0.1, 0.3, 0.5],
    'weight_sentiment': [0.1, 0.3, 0.5],
    
    # --- Trade Execution Parameters ---
    'trade_entry_threshold': [0.4, 0.6, 0.8], # Entry threshold for confidence score
    'sl_atr_multiplier': [1.0, 2.0, 3.0], # Stop loss distance in multiples of ATR
    'take_profit_rr': [1.5, 2.5, 3.5], # Risk/Reward ratio for setting the take profit

    # --- Key Indicator Parameters (Periods, Thresholds, Multipliers) ---
    'adx_period': [10, 20, 30], # ADX period for trend strength
    'zscore_threshold': [1.0, 2.0, 3.0], # Z-score threshold for volume delta
    'obv_lookback': [15, 30, 60], # OBV lookback period
    'bband_length': [15, 30, 60], # Bollinger Band length
    'bband_squeeze_threshold': [0.005, 0.015, 0.025], # Bollinger Band squeeze threshold
    'atr_period': [10, 14, 20], # ATR period
    'proximity_multiplier': [0.1, 0.25, 0.4], # Proximity multiplier for S/R interaction
    'sma_period': [30, 50, 70], # SMA period for mean reversion
    'volume_multiplier': [2, 3, 4], # Volume multiplier for regime labeling
    'volatility_multiplier': [1, 2, 3], # Volatility multiplier for regime labeling
    'trend_threshold': [20, 25, 30], # ADX trend threshold
    'max_strength_threshold': [50, 60, 70], # ADX max strength threshold
    'bband_std': [1.5, 2.0, 2.5], # Bollinger Band standard deviation
    'scaling_factor': [50, 100, 150], # Scaling factor for MACD histogram normalization
    'trend_strength_threshold': [20, 25, 30] # Threshold for trend strength in regime labeling
}

NUM_TUNING_POINTS = 5 
TUNING_PERCENTAGE = 0.2 
INTEGER_PARAMS = [
    'adx_period', 'atr_period', 'sma_period', 'obv_lookback', 'bband_length',
    'volume_multiplier', 'volatility_multiplier', 'trend_threshold', 
    'max_strength_threshold', 'scaling_factor', 'trend_strength_threshold',
]

TOP_N_RESULTS = 5

def run_grid_search_stage(param_grid, klines_df, agg_trades_df, futures_df, sr_levels, stage_name):
    """Runs a full grid search for a given parameter grid."""
    print(f"\n--- Starting {stage_name} ---")
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(param_combinations)} combinations for this stage...")
    
    results = []
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
        
        # Pass sr_levels to calculate_features_and_score
        prepared_df = calculate_features_and_score(
            klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), params, sr_levels
        )
        
        portfolio_result = run_backtest(prepared_df, params)
        
        if len(portfolio_result.trades) == 0:
            print("--> NO TRADES WERE MADE.")
        else:
            print(f"--> Resulting Equity: ${portfolio_result.equity:,.2f} | Total Trades: {len(portfolio_result.trades)}")
        
        results.append({'params': params, 'portfolio': portfolio_result})
        
    results.sort(key=lambda x: x['portfolio'].equity, reverse=True)
    return results

def run_coordinate_descent_stage(best_params, klines_df, agg_trades_df, futures_df, sr_levels):
    """Performs Stage 2: Fine-tunes each parameter one by one."""
    print("\n--- Starting Stage 2: Coordinate Descent Fine-Tuning ---")
    
    current_best_params = copy.deepcopy(best_params)
    
    # Iterate through each parameter that was in the coarse grid
    for param_to_tune in COARSE_PARAM_GRID.keys():
        print(f"\n--- Tuning Parameter: '{param_to_tune}' ---")
        
        best_value = current_best_params[param_to_tune]
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
            test_params = current_best_params.copy()
            test_params[param_to_tune] = value

            # Pass sr_levels to calculate_features_and_score
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
            current_best_params[param_to_tune] = best_value_for_param

    print("\n--- Running Final Backtest with Tuned Parameters ---")
    # Pass sr_levels to calculate_features_and_score
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
    report_lines.append("RECOMMENDATION: Update the BEST_PARAMS dictionary in config.py with the final tuned values.")
    report_lines.append(separator)
    return "\n".join(report_lines)

# This main block is now for running the optimizer as a standalone script
if __name__ == "__main__":
    email_subject = "Ares Optimizer FAILED"
    email_body = "The script encountered an unexpected error."
    try:
        print("--- Ares Efficient Strategy Optimizer (Standalone) ---")
        klines_df, agg_trades_df, futures_df = load_raw_data() # Load futures_df here
        daily_df = klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        sr_levels = get_sr_levels(daily_df)
        # Pass futures_df to the optimization stages
        coarse_results = run_grid_search_stage(COARSE_PARAM_GRID, klines_df, agg_trades_df, futures_df, sr_levels, "Stage 1: Coarse Grid Search")
        if not coarse_results or coarse_results[0]['portfolio'].equity <= INITIAL_EQUITY:
            print("\nCoarse search did not yield any profitable results. Exiting.")
            email_subject = "Ares Optimizer Finished (No Profitable Results)"
            email_body = "The coarse search stage did not find any parameter combinations that resulted in a profit."
        else:
            best_coarse_params = coarse_results[0]['params']
            print(f"\nBest result from Coarse Search: ${coarse_results[0]['portfolio'].equity:,.2f}")
            print(f"Best Coarse Params: {best_coarse_params}")
            # Pass futures_df to the optimization stages
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
