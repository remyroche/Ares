import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data(directory):
    """
    Loads all formatted CSV files from a directory into a single, sorted DataFrame.

    Args:
        directory (str): The path to the directory containing formatted CSV files.

    Returns:
        pandas.DataFrame: A DataFrame with all trade data, sorted by timestamp.
                          Returns None if the directory is not found or is empty.
    """
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('formatted_') and f.endswith('.csv')]
    
    if not all_files:
        print(f"Error: No formatted CSV files found in '{directory}'.")
        print("Please run the initial formatting script first.")
        return None

    print(f"Loading {len(all_files)} files from '{directory}'...")
    
    df_list = []
    for f in all_files:
        try:
            # Read the CSV, making sure to parse the timestamp column correctly.
            temp_df = pd.read_csv(f, parse_dates=['timestamp'])
            df_list.append(temp_df)
        except Exception as e:
            print(f"Warning: Could not read or process file {f}. Error: {e}")
            continue
    
    if not df_list:
        print("Error: No data could be loaded. Aborting.")
        return None

    # Concatenate all dataframes into one
    full_df = pd.concat(df_list, ignore_index=True)

    # Convert price and quantity to numeric types, coercing errors to NaN
    full_df['price'] = pd.to_numeric(full_df['price'], errors='coerce')
    full_df['quantity'] = pd.to_numeric(full_df['quantity'], errors='coerce')

    # Drop any rows that have missing data after conversion
    full_df.dropna(subset=['timestamp', 'price'], inplace=True)

    # Sort the entire dataset by timestamp to ensure chronological order
    full_df.sort_values('timestamp', inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    print(f"Successfully loaded and prepared {len(full_df)} total trades.")
    return full_df

def analyze_price_action(df, target_pct, stop_loss_pct):
    """
    Calculates the average time and number of occurrences for hitting a price 
    target without hitting a stop loss.

    Args:
        df (pandas.DataFrame): The DataFrame with trade data.
        target_pct (float): The target profit percentage (e.g., 0.5 for 0.5%).
        stop_loss_pct (float): The stop loss percentage (e.g., 0.1 for 0.1%).

    Returns:
        tuple: A tuple containing:
               - float: The average time in seconds to hit the target.
               - int: The number of times the event occurred.
               Returns (NaN, 0) if no such events are found.
    """
    durations = []
    
    # Convert percentages to multipliers for calculation
    upper_target_multiplier = 1 + (target_pct / 100)
    lower_target_multiplier = 1 - (target_pct / 100)
    
    upper_stop_multiplier = 1 + (stop_loss_pct / 100)
    lower_stop_multiplier = 1 - (stop_loss_pct / 100)

    # Iterate through each trade as a potential starting point
    for i in range(len(df)):
        start_price = df.at[i, 'price']
        start_time = df.at[i, 'timestamp']
        
        # Define the price levels for our targets and stops
        up_target_price = start_price * upper_target_multiplier
        down_target_price = start_price * lower_target_multiplier
        
        up_stop_price = start_price * upper_stop_multiplier
        down_stop_price = start_price * lower_stop_multiplier

        # Look at all subsequent trades to find when a barrier is hit
        future_trades = df.iloc[i+1:]

        # --- Check for an upward move ---
        up_target_hit_trades = future_trades[future_trades['price'] >= up_target_price]
        if not up_target_hit_trades.empty:
            first_up_hit_index = up_target_hit_trades.index[0]
            trades_before_up_hit = future_trades.loc[:first_up_hit_index]
            if trades_before_up_hit[trades_before_up_hit['price'] <= down_stop_price].empty:
                end_time = df.at[first_up_hit_index, 'timestamp']
                duration = (end_time - start_time).total_seconds()
                durations.append(duration)

        # --- Check for a downward move ---
        down_target_hit_trades = future_trades[future_trades['price'] <= down_target_price]
        if not down_target_hit_trades.empty:
            first_down_hit_index = down_target_hit_trades.index[0]
            trades_before_down_hit = future_trades.loc[:first_down_hit_index]
            if trades_before_down_hit[trades_before_down_hit['price'] >= up_stop_price].empty:
                end_time = df.at[first_down_hit_index, 'timestamp']
                duration = (end_time - start_time).total_seconds()
                durations.append(duration)

    if not durations:
        return np.nan, 0
    
    return np.mean(durations), len(durations)


if __name__ == '__main__':
    # --- Configuration ---
    FORMATTED_DATA_DIR = "formatted_csvs"
    
    target_percentages = np.arange(0.5, 1.6, 0.1)
    stop_loss_percentages = np.arange(0.1, 1.1, 0.1)

    # --- Execution ---
    trade_data = load_and_prepare_data(FORMATTED_DATA_DIR)

    if trade_data is not None and not trade_data.empty:
        # Create DataFrames to store the results for time and counts
        index_labels = [f"{p:.1f}%" for p in target_percentages]
        column_labels = [f"{p:.1f}%" for p in stop_loss_percentages]

        results_time_df = pd.DataFrame(index=index_labels, columns=column_labels, dtype=float)
        results_time_df.index.name = "Target %"
        results_time_df.columns.name = "Stop Loss %"

        results_counts_df = pd.DataFrame(index=index_labels, columns=column_labels, dtype=int)
        results_counts_df.index.name = "Target %"
        results_counts_df.columns.name = "Stop Loss %"

        print("\n--- Starting Price Action Analysis ---")
        print("This may take some time depending on the size of your dataset...")
        
        for target in target_percentages:
            for stop_loss in stop_loss_percentages:
                avg_time, count = analyze_price_action(trade_data, target, stop_loss)
                
                # Store both results in their respective tables
                results_time_df.loc[f"{target:.1f}%", f"{stop_loss:.1f}%"] = avg_time
                results_counts_df.loc[f"{target:.1f}%", f"{stop_loss:.1f}%"] = count
                
                print(f"  - Target: {target:.1f}%, Stop: {stop_loss:.1f}% -> Avg Time: {avg_time:.2f}s, Occurrences: {count}" if not np.isnan(avg_time) else f"  - Target: {target:.1f}%, Stop: {stop_loss:.1f}% -> No events found")

        # Display the final results tables
        print("\n" + "="*50)
        print("                 Analysis Complete")
        print("="*50)
        
        print("\nAverage time (in seconds) to hit target:")
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(results_time_df)
        
        print("\n" + "="*50)
        print("\nNumber of occurrences for each scenario:")
        print(results_counts_df)
