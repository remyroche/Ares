import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
import os
import subprocess
import sys
# Import centralized data loading functions
from src.analyst.data_utils import load_klines_data, load_agg_trades_data, load_futures_data
from config import (
    KLINES_FILENAME, AGG_TRADES_FILENAME, FUTURES_FILENAME, PREPARED_DATA_FILENAME, 
    BEST_PARAMS, DOWNLOADER_SCRIPT_NAME, INTERVAL
)

# --- DEBUGGING FLAG ---
DEBUG_MODE = True

def load_raw_data():
    """Loads raw data, now including futures data."""
    print("--- Step 1: Loading Raw Data ---")
    required_files = [KLINES_FILENAME, AGG_TRADES_FILENAME, FUTURES_FILENAME]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing raw data files: {', '.join(missing_files)}")
        print(f"Calling downloader script: '{DOWNLOADER_SCRIPT_NAME}'...")
        try:
            subprocess.run([sys.executable, DOWNLOADER_SCRIPT_NAME], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"ERROR during download: {e}")
            sys.exit(1)
            
    try:
        # Use the centralized data loading functions
        klines_df = load_klines_data(KLINES_FILENAME)
        agg_trades_df = load_agg_trades_data(AGG_TRADES_FILENAME)
        futures_df = load_futures_data(FUTURES_FILENAME)
        
        # Duplicated index handling is now done within the centralized load functions,
        # but keeping these lines for robustness in case data source is not perfectly clean.
        klines_df = klines_df[~klines_df.index.duplicated(keep='first')]
        agg_trades_df = agg_trades_df[~agg_trades_df.index.duplicated(keep='first')]
        
        print(f"Loaded {len(klines_df)} k-lines, {len(agg_trades_df)} agg trades, and {len(futures_df)} futures data points.\n")
        return klines_df, agg_trades_df, futures_df
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required file: {e}. Exiting.")
        sys.exit(1)

def get_sr_levels(df_daily):
    """Identifies Support/Resistance levels from daily aggregated data."""
    print("--- Step 2: Identifying S/R Levels ---")
    recent_daily = df_daily.tail(365)
    price_bins = pd.cut(recent_daily['close'], bins=100)
    volume_profile = recent_daily.groupby(price_bins, observed=False)['volume'].sum()
    hvn_levels = volume_profile.nlargest(5).index.map(lambda x: x.mid).tolist()
    poc = volume_profile.idxmax().mid
    high_peaks, _ = find_peaks(recent_daily['high'], distance=15)
    low_peaks, _ = find_peaks(-recent_daily['low'], distance=15)
    pivots = recent_daily.iloc[high_peaks]['high'].tolist() + recent_daily.iloc[low_peaks]['low'].tolist()
    levels = pd.DataFrame({'price': hvn_levels + [poc] + pivots}).drop_duplicates()
    print("S/R Level identification complete.\n")
    return levels['price'].tolist()


def calculate_features_and_score(klines_df, agg_trades_df, futures_df, params, sr_levels):
    """
    Calculates all feature sub-scores and combines them into a final Confidence Score.
    Now accepts sr_levels and uses params for all indicator settings.
    """
    print("--- Step 3: Calculating Features and Confidence Score ---")
    
    klines_df = pd.merge_asof(klines_df.sort_index(), futures_df.sort_index(), left_index=True, right_index=True, direction='backward')
    klines_df.fillna(method='ffill', inplace=True)

    # --- 1. Calculate Sub-Scores for each component ---
    
    # Sentiment Score
    oi_change = klines_df['openInterest'].pct_change().rolling(window=240).mean()
    oi_score = (oi_change - oi_change.mean()) / oi_change.std()
    fr_score = (klines_df['fundingRate'] - klines_df['fundingRate'].mean()) / klines_df['fundingRate'].std()
    klines_df['sentiment_score'] = np.clip((oi_score + fr_score) / 2, -1, 1).fillna(0)

    # Trend Score (using ADX and MACD)
    klines_df.ta.adx(length=params['adx_period'], append=True, col_names=('ADX', 'DMP', 'DMN'))
    macd = klines_df.ta.macd(append=False)
    klines_df['MACD_HIST'] = macd['MACDh_12_26_9']
    normalized_hist = klines_df['MACD_HIST'] / klines_df['close']
    direction_score = np.clip(normalized_hist * params['scaling_factor'], -1, 1)
    momentum_score = np.clip((klines_df['ADX'] - params['trend_threshold']) / 
                              (params['max_strength_threshold'] - params['trend_threshold']), 0, 1)
    klines_df['trend_score'] = (direction_score * momentum_score).fillna(0)

    # Mean Reversion Score (using Delta Z-Score at S/R levels)
    klines_df.ta.atr(length=params['atr_period'], append=True, col_names=('ATR'))
    
    # Calculate Is_Interacting based on proximity to SR levels
    proximity_threshold = klines_df['ATR'] * params['proximity_multiplier']
    is_interacting = pd.Series(False, index=klines_df.index)
    for level in sr_levels:
        is_interacting = is_interacting | ((klines_df['low'] <= level + proximity_threshold) & 
                                            (klines_df['high'] >= level - proximity_threshold))
    klines_df['Is_Interacting'] = is_interacting.astype(int) # Convert boolean to 0/1

    agg_trades_df['delta'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
    klines_df['volume_delta'] = agg_trades_df['delta'].resample(INTERVAL).sum().fillna(0)
    delta_mean = klines_df['volume_delta'].rolling(window=60).mean()
    delta_std = klines_df['volume_delta'].rolling(window=60).std()
    klines_df['delta_zscore'] = ((klines_df['volume_delta'] - delta_mean) / delta_std.replace(0, np.nan)).fillna(0)
    # A high positive z-score is bearish absorption, so we invert it for the score
    klines_df['reversion_score'] = np.clip(-klines_df['delta_zscore'] / params['zscore_threshold'], -1, 1) * klines_df['Is_Interacting']

    # --- 2. Combine Sub-Scores into a Final Confidence Score using Weights ---
    klines_df['Confidence_Score'] = (
        klines_df['trend_score'] * params['weight_trend'] +
        klines_df['reversion_score'] * params['weight_reversion'] +
        klines_df['sentiment_score'] * params['weight_sentiment']
    )
    
    # Normalize the final score to be between -1 and 1
    total_weight = params['weight_trend'] + params['weight_reversion'] + params['weight_sentiment']
    if total_weight > 0:
        klines_df['Confidence_Score'] /= total_weight
    
    # Drop rows with NaNs from indicator calculations
    final_df = klines_df.dropna(subset=['ATR', 'Confidence_Score'])
    
    if DEBUG_MODE:
        print("\n--- DEBUG INFO (Post-Scoring) ---")
        print("Confidence Score Distribution:")
        print(final_df['Confidence_Score'].describe())
        print(f"\nTotal rows sent to backtester: {len(final_df)}")
        print("--- END DEBUG INFO ---\n")
    
    print("Feature calculation and scoring complete.\n")
    return final_df

def calculate_and_label_regimes(klines_df, agg_trades_df, futures_df, params, sr_levels, trend_strength_threshold_for_regimes=None):
    """
    Calculates features and labels market regimes based on the provided parameters.
    This function now explicitly labels a 'Is_Strong_Trend' regime.
    """
    print("--- Step X: Calculating and Labeling Regimes ---")

    # Ensure the params dictionary contains the 'trend_strength_threshold'
    # If trend_strength_threshold_for_regimes is provided (from older calls), use it as fallback.
    # Otherwise, it should be in params.
    if 'trend_strength_threshold' not in params and trend_strength_threshold_for_regimes is not None:
        params['trend_strength_threshold'] = trend_strength_threshold_for_regimes
    elif 'trend_strength_threshold' not in params:
        # Fallback to a default if not in params and not provided separately
        params['trend_strength_threshold'] = 25 
        print(f"Warning: 'trend_strength_threshold' not found in params. Using default: {params['trend_strength_threshold']}")


    # First, calculate the core features and confidence score
    prepared_df = calculate_features_and_score(klines_df, agg_trades_df, futures_df, params, sr_levels)
    
    # Add regime labeling logic here
    # Example: Labeling a "Strong Trend" regime based on ADX and a threshold
    # Ensure ADX is calculated in calculate_features_and_score before this step
    if 'ADX' in prepared_df.columns:
        prepared_df['Is_Strong_Trend'] = (prepared_df['ADX'] > params['trend_strength_threshold']).astype(int)
        print(f"Labeled 'Is_Strong_Trend' regime using ADX > {params['trend_strength_threshold']}.")
    else:
        print("ADX not found in prepared_df. Skipping 'Is_Strong_Trend' labeling.")
        prepared_df['Is_Strong_Trend'] = 0 # Default to no strong trend if ADX is missing

    print("Regime labeling and feature calculation complete.\n")
    return prepared_df


if __name__ == "__main__":
    # Ensure dummy data files exist for backtesting
    # Moved create_dummy_data to src/analyst/data_utils.py
    from src.analyst.data_utils import create_dummy_data
    from config import KLINES_FILENAME, AGG_TRADES_FILENAME, FUTURES_FILENAME

    create_dummy_data(KLINES_FILENAME, 'klines')
    create_dummy_data(AGG_TRADES_FILENAME, 'agg_trades')
    create_dummy_data(FUTURES_FILENAME, 'futures')

    klines_df, agg_trades_df, futures_df = load_raw_data()
    if klines_df is None: sys.exit(1)
    daily_df = klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    sr_levels = get_sr_levels(daily_df)
    
    # When running standalone, use BEST_PARAMS from config.py
    # Ensure BEST_PARAMS has 'trend_strength_threshold' or provide a default
    if 'trend_strength_threshold' not in BEST_PARAMS:
        BEST_PARAMS['trend_strength_threshold'] = 25 # Default value if not in BEST_PARAMS

    # Pass all necessary arguments to calculate_and_label_regimes
    prepared_df = calculate_and_label_regimes(klines_df, agg_trades_df, futures_df, BEST_PARAMS, sr_levels, BEST_PARAMS['trend_strength_threshold'])
    prepared_df.to_csv(PREPARED_DATA_FILENAME)
    print(f"--- Final Prepared Data Saved to '{PREPARED_DATA_FILENAME}' ---")
