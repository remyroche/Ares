# analyze_timeframe.py

# This Python script analyzes historical trade data to determine the average time required to reach a specific net profit target without first hitting a stop-loss. 
# By calculating a score that prioritizes frequent and fast outcomes, it identifies the most efficient combination of profit and risk parameters.
# After accounting for Binance trading fees, to help define an optimal timeframe for market labeling.

def load_and_prepare_data(directory):
    """
    Loads all formatted CSV files, then resamples the data to a 1-second frequency.

    Args:
        directory (str): The path to the directory containing formatted CSV files.

    Returns:
        pandas.DataFrame: A DataFrame with price data at 1-second intervals,
                          sorted by timestamp. Returns None if the directory is
                          not found or is empty.
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
            temp_df = pd.read_csv(f, parse_dates=['timestamp'])
            df_list.append(temp_df)
        except Exception as e:
            print(f"Warning: Could not read or process file {f}. Error: {e}")
            continue
    
    if not df_list:
        print("Error: No data could be loaded. Aborting.")
        return None

    full_df = pd.concat(df_list, ignore_index=True)
    full_df['price'] = pd.to_numeric(full_df['price'], errors='coerce')
    full_df.dropna(subset=['timestamp', 'price'], inplace=True)
    full_df.sort_values('timestamp', inplace=True)

    print(f"Successfully loaded {len(full_df)} total trades.")
    
    print("Resampling data to a 1-second frequency...")
    if full_df.empty:
        print("Warning: No data available to resample.")
        return full_df

    full_df.set_index('timestamp', inplace=True)
    resampled_df = full_df['price'].resample('1S').last().ffill()
    resampled_df = resampled_df.reset_index()
    
    print(f"Data has been resampled to {len(resampled_df)} one-second intervals.")
    
    return resampled_df

def analyze_price_action(df, net_target_pct, stop_loss_pct, round_trip_fee_pct):
    """
    Calculates the average time and occurrences for hitting a *net* price target
    (after fees) without hitting a stop loss.

    Args:
        df (pandas.DataFrame): The DataFrame with 1-second interval trade data.
        net_target_pct (float): The desired *net* profit percentage.
        stop_loss_pct (float): The stop loss percentage on gross price movement.
        round_trip_fee_pct (float): The total fee percentage for opening and closing.

    Returns:
        tuple: (average_time_seconds, number_of_occurrences)
    """
    durations = []
    
    # The price must move by the net target PLUS the fees to be profitable.
    gross_target_pct = net_target_pct + round_trip_fee_pct
    
    upper_target_multiplier = 1 + (gross_target_pct / 100)
    lower_target_multiplier = 1 - (gross_target_pct / 100)
    upper_stop_multiplier = 1 + (stop_loss_pct / 100)
    lower_stop_multiplier = 1 - (stop_loss_pct / 100)

    prices = df['price'].values
    timestamps = df['timestamp'].values

    for i in range(len(df)):
        start_price = prices[i]
        start_time = timestamps[i]
        
        up_target_price = start_price * upper_target_multiplier
        down_target_price = start_price * lower_target_multiplier
        up_stop_price = start_price * upper_stop_multiplier
        down_stop_price = start_price * lower_stop_multiplier

        for j in range(i + 1, len(df)):
            current_price = prices[j]
            
            hit_up_target = current_price >= up_target_price
            hit_down_target = current_price <= down_target_price
            hit_up_stop = current_price >= up_stop_price
            hit_down_stop = current_price <= down_stop_price

            if hit_up_target:
                path_prices = prices[i+1:j+1]
                if not np.any(path_prices <= down_stop_price):
                    duration = (timestamps[j] - start_time).total_seconds()
                    durations.append(duration)
                break 

            if hit_down_target:
                path_prices = prices[i+1:j+1]
                if not np.any(path_prices >= up_stop_price):
                    duration = (timestamps[j] - start_time).total_seconds()
                    durations.append(duration)
                break
            
            if hit_down_stop or hit_up_stop:
                break

    if not durations:
        return np.nan, 0
    
    return np.mean(durations), len(durations)


if __name__ == '__main__':
    # --- Configuration ---
    FORMATTED_DATA_DIR = "formatted_csvs"
    OUTPUT_CSV_FILENAME = "price_action_analysis.csv"
    
    # --- Fee Configuration (Binance USDT-M Futures, Non-VIP) ---
    ROUND_TRIP_FEE_PCT = 0.08
    print(f"Using a conservative round-trip fee of {ROUND_TRIP_FEE_PCT}% for all calculations.")

    # These are your desired NET profit targets
    target_percentages = np.arange(0.5, 1.6, 0.1)
    stop_loss_percentages = np.arange(0.1, 1.1, 0.1)

    # --- Execution ---
    trade_data = load_and_prepare_data(FORMATTED_DATA_DIR)

    if trade_data is not None and not trade_data.empty:
        index_labels = [f"{p:.1f}%" for p in target_percentages]
        column_labels = [f"{p:.1f}%" for p in stop_loss_percentages]

        results_display_df = pd.DataFrame(index=index_labels, columns=column_labels, dtype=object)
        results_display_df.index.name = "Net Target %"
        results_display_df.columns.name = "Stop Loss %"

        results_score_df = pd.DataFrame(index=index_labels, columns=column_labels, dtype=float)

        print("\n--- Starting Price Action Analysis (1-Second Intervals, Fees Included) ---")
        
        for net_target in target_percentages:
            for stop_loss in stop_loss_percentages:
                avg_time, count = analyze_price_action(trade_data, net_target, stop_loss, ROUND_TRIP_FEE_PCT)
                
                score = 0
                display_str = "N/A"

                if count > 0 and not np.isnan(avg_time) and avg_time > 0:
                    score = (count ** 2) / avg_time
                    display_str = f"{avg_time:.1f}s | {count} | {score:.2f}"

                results_display_df.loc[f"{net_target:.1f}%", f"{stop_loss:.1f}%"] = display_str
                results_score_df.loc[f"{net_target:.1f}%", f"{stop_loss:.1f}%"] = score
                
                print(f"  - Net Target: {net_target:.1f}%, Stop: {stop_loss:.1f}% -> {display_str}")

        # --- Display and Save Results ---
        print("\n" + "="*70)
        print("                     Analysis Complete")
        print("="*70)
        
        print("\nCombined Results (Avg Time | Occurrences | Freq-to-Time Score):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print(results_display_df)

        # Save the results DataFrame to a CSV file
        try:
            results_display_df.to_csv(OUTPUT_CSV_FILENAME)
            print(f"\nResults have been successfully saved to '{OUTPUT_CSV_FILENAME}'")
        except Exception as e:
            print(f"\nError: Could not save results to CSV. Reason: {e}")

        # --- Find and print the best score ---
        max_score = results_score_df.max().max()
        if max_score > 0:
            max_pos = results_score_df.stack().idxmax()
            ideal_target, ideal_stop = max_pos
            
            print("\n" + "="*70)
            print("                       Ideal Parameters (Net Profit)")
            print("="*70)
            print(f"\nThe combination with the highest Frequency-to-Time Score ({max_score:.2f}) is:")
            print(f"  - Desired Net Profit: {ideal_target}")
            print(f"  - Stop Loss:          {ideal_stop}")
            print("\nThis combination offers the best balance of high frequency and fast resolution after fees.")
        else:
            print("\nCould not determine an ideal parameter set as no valid events were found.")
